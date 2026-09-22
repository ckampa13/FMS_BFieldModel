'''
Build n-pole (multipole) accelerator magnets out of straight rectangular bars
that helicalc's StraightIntegrator3D can integrate.

The truth field is therefore Biot-Savart from real conductor geometry rather than
a prescribed analytic model.  See CLAUDE/HANDOFF_helicalc_multipoles.md for the
physics brief and the target HL-LHC-like assembly.

Conventions (must match helicalc/busbar.py):
  - right-handed cartesian (x, y, z), beam axis +z; phi = atan2(y, x)
  - multipole order n: 1 dipole, 2 quadrupole, 3 sextupole, ... (European)
  - strength quoted as B_n(R_ref), the transverse field magnitude at R_ref:
        normal:  B_y + i B_x =     B_n(R_ref) * ((x+iy)/R_ref)**(n-1)
        skew:    B_y + i B_x = i * A_n(R_ref) * ((x+iy)/R_ref)**(n-1)
  - metres, amperes, tesla internally.

This module is pure numpy/pandas -- it does NOT import torch, so the geometry can
be built and checked on a machine with no GPU.  The field aggregator that does
need torch lives in multipole_field.py.
'''
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from .constants import mu0

# exact Mu2e_Straight_Bars_V*.csv schema, so generated geometry is a drop-in
# for StraightIntegrator3D and for solenoid_geom_funcs.get_3d_straight
STRAIGHT_COLS = ['Name/role', 'cond N', 'R0', 'phi0', 'R1', 'phi1', 'dphi',
                 'length', "x0'", 'x0', 'y0', 'z0', "x1'", 'x1', 'y1',
                 'xmid', 'ymid', 'z_mid', 'W', 'T', 'I', 'J', 'I_flow',
                 'alpha', 'Phi2', 'theta2', 'psi2']

# extra bookkeeping columns we add (ignored by the integrator)
EXTRA_COLS = ['element', 'n', 'skew', 'seg_kind', 'loop']


# ----------------------------------------------------------------------------
# geometry parameters
# ----------------------------------------------------------------------------
@dataclass
class MultipoleGeom:
    '''Physical dimensions of a multipole winding.

    `aperture` (the DIAMETER of the clear bore) is the primary input; everything
    else derives from it unless given explicitly, so changing the aperture
    rescales the winding radius, the clearance, the harmonic reference radius,
    every current and the end-turn segmentation consistently.

    Defaults reproduce the HL-LHC-like assembly of the handoff: 150 mm aperture
    -> R_aper = 75 mm, R_ref = 50 mm, R_map = 60 mm, r_clear = 75 mm.  Note the
    default winding radius is `R_aper + T/2` = 82.5 mm (bar inner face flush with
    the aperture wall); pass a=0.090 for the handoff's nominal value.
    '''
    aperture: float = 0.150     # aperture DIAMETER [m]
    W: float = 0.008            # bar cross-section, azimuthal [m]
    T: float = 0.015            # bar cross-section, radial [m]
    a: float = None             # winding mean radius [m]
    R_ref: float = None         # harmonic reference radius [m]
    R_map: float = None         # mapping / measurement radius [m]
    r_clear: float = None       # min radius any conductor may approach [m]
    N_target: int = 32          # bars per winding, before rounding to a multiple of 4n

    @property
    def R_aper(self):
        return 0.5 * self.aperture

    def __post_init__(self):
        R = self.R_aper
        if self.a is None:
            self.a = R + 0.5 * self.T      # inner face flush with the aperture wall
        if self.R_ref is None:
            self.R_ref = R / 1.5           # 50 mm for a 150 mm aperture
        if self.R_map is None:
            self.R_map = 0.8 * R           # 60 mm for a 150 mm aperture
        if self.r_clear is None:
            self.r_clear = R
        self.validate()

    def validate(self):
        if self.aperture <= 0 or self.W <= 0 or self.T <= 0:
            raise ValueError('aperture, W and T must be positive')
        if self.a - 0.5 * self.T < self.R_aper - 1e-12:
            raise ValueError(
                'bar inner face (r=%.4f m) is inside the aperture wall (r=%.4f m); '
                'increase `a` or decrease `T`' % (self.a - 0.5 * self.T, self.R_aper))
        if self.R_map >= self.R_aper:
            raise ValueError('R_map (%.4f m) must be inside the aperture (%.4f m)'
                             % (self.R_map, self.R_aper))
        if self.R_ref > self.R_map + 1e-12:
            raise ValueError('R_ref (%.4f m) should not exceed R_map (%.4f m)'
                             % (self.R_ref, self.R_map))
        if self.r_clear < self.R_aper - 1e-12:
            raise ValueError('r_clear (%.4f m) would let end turns cross the '
                             'aperture wall (%.4f m)' % (self.r_clear, self.R_aper))
        if self.r_clear > self.a + 1e-12:
            raise ValueError('r_clear (%.4f m) exceeds the winding radius (%.4f m)'
                             % (self.r_clear, self.a))
        return self

    def with_(self, **kwargs):
        '''Copy with overrides, re-deriving any field that was not set explicitly.'''
        derived = ('a', 'R_ref', 'R_map', 'r_clear')
        # drop derived values unless they (or aperture-independent inputs) are
        # being overridden, so e.g. .with_(aperture=0.10) rescales everything
        base = {f: getattr(self, f) for f in
                ('aperture', 'W', 'T', 'N_target') + derived}
        if 'aperture' in kwargs:
            for f in derived:
                if f not in kwargs:
                    base[f] = None
        base.update(kwargs)
        return MultipoleGeom(**base)


def _resolve_geom(geom, overrides):
    if geom is None:
        geom = MultipoleGeom()
    return geom.with_(**overrides) if overrides else geom


# ----------------------------------------------------------------------------
# single-bar primitive
# ----------------------------------------------------------------------------
def bar_angles(p0, p1, psi_mode='radial'):
    '''ZYZ Euler angles (degrees) placing a helicalc straight bar from p0 to p1.

    The bar's local +z' runs from p0 to p1 (current direction), so
        theta2 = acos(dz/L),  Phi2 = atan2(dy, dx).
    psi2 only rolls the rectangular cross-section about the bar axis:
      'radial' -- T along the cylindrical radial direction, W azimuthal.  This is
                  what an axial winding bar on the cylinder r=a wants.
      'normal' -- T along the local z axis component normal to the bar, i.e. the
                  cross-section's T direction is pushed out of the transverse
                  plane.  Used for end-turn segments, which lie at constant z.
    Returns (length, Phi2, theta2, psi2) with angles in degrees.
    '''
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L <= 0:
        raise ValueError('zero-length bar: p0 == p1')
    dhat = d / L
    theta2 = np.degrees(np.arccos(np.clip(dhat[2], -1.0, 1.0)))
    Phi2 = np.degrees(np.arctan2(dhat[1], dhat[0]))

    # target direction for the local y' axis (the T direction)
    mid = 0.5 * (p0 + p1)
    if psi_mode == 'radial':
        t_tgt = np.array([mid[0], mid[1], 0.0])
    elif psi_mode == 'normal':
        # normal to the plane containing the bar and the z axis
        t_tgt = np.cross(dhat, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(t_tgt) < 1e-12:          # bar is axial
            t_tgt = np.array([mid[0], mid[1], 0.0])
    else:
        raise ValueError("psi_mode must be 'radial' or 'normal'")

    # project the target into the plane normal to the bar
    t_tgt = t_tgt - np.dot(t_tgt, dhat) * dhat
    if np.linalg.norm(t_tgt) < 1e-12:
        t_tgt = np.array([1.0, 0.0, 0.0])
        t_tgt = t_tgt - np.dot(t_tgt, dhat) * dhat
    t_tgt /= np.linalg.norm(t_tgt)

    # R = Rz(Phi2) Ry(theta2) Rz(psi2), so the global image of local y' is
    #     y_hat = -sin(psi2) * x0_hat + cos(psi2) * y0_hat
    # where x0_hat, y0_hat are the images at psi2 = 0.  Setting y_hat = t_tgt:
    cP, sP = np.cos(np.radians(Phi2)), np.sin(np.radians(Phi2))
    cT, sT = np.cos(np.radians(theta2)), np.sin(np.radians(theta2))
    x0_hat = np.array([cP * cT, sP * cT, -sT])     # R(psi2=0) applied to local x'
    y0_hat = np.array([-sP, cP, 0.0])              # R(psi2=0) applied to local y'
    psi2 = np.degrees(np.arctan2(-np.dot(t_tgt, x0_hat), np.dot(t_tgt, y0_hat)))
    return L, Phi2, theta2, psi2


def make_bar_row(p0, p1, I, cond_N, geom=None, W=None, T=None, name='',
                 psi_mode='radial', element='', n=0, skew=False,
                 seg_kind='axial', loop=-1):
    '''One straight bar, as a dict in the Mu2e straight-bar CSV schema.

    Current flows p0 -> p1 with signed magnitude I (negative I flips the field,
    which is how cos(n.theta) sign changes are encoded).  I_flow is always 0, so
    helicalc takes (x0, y0, z0) as the local origin.
    '''
    if W is None or T is None:
        g = geom if geom is not None else MultipoleGeom()
        W = g.W if W is None else W
        T = g.T if T is None else T
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    L, Phi2, theta2, psi2 = bar_angles(p0, p1, psi_mode=psi_mode)
    mid = 0.5 * (p0 + p1)
    return {
        'Name/role': name, 'cond N': int(cond_N),
        'R0': float(np.hypot(p0[0], p0[1])),
        'phi0': float(np.degrees(np.arctan2(p0[1], p0[0]))),
        'R1': float(np.hypot(p1[0], p1[1])),
        'phi1': float(np.degrees(np.arctan2(p1[1], p1[0]))),
        'dphi': np.nan, 'length': L,
        "x0'": np.nan, 'x0': p0[0], 'y0': p0[1], 'z0': p0[2],
        "x1'": np.nan, 'x1': p1[0], 'y1': p1[1],
        'xmid': mid[0], 'ymid': mid[1], 'z_mid': mid[2],
        'W': W, 'T': T, 'I': I, 'J': I / (W * T), 'I_flow': 0,
        'alpha': np.nan, 'Phi2': Phi2, 'theta2': theta2, 'psi2': psi2,
        # bookkeeping
        'element': element, 'n': n, 'skew': bool(skew),
        'seg_kind': seg_kind, 'loop': loop,
        # true bar endpoint, kept because the CSV's x1/y1 carry no z
        'z1': p1[2],
    }


def bars_to_df(rows):
    '''DataFrame in schema order (plus bookkeeping columns) from make_bar_row dicts.'''
    df = pd.DataFrame(rows)
    cols = [c for c in STRAIGHT_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]


# ----------------------------------------------------------------------------
# currents and bar counts
# ----------------------------------------------------------------------------
def current_for_strength(n, B_ref, N, geom=None, b=None, **geom_overrides):
    '''Bar current amplitude I0 giving B_n(R_ref) = B_ref.

    For N axial bars at radius a with I_i = I0 * cos(n.theta_i), the interior
    field is a pure 2n-pole with |B_n(R_ref)| = mu0*N*|I0|/(4*pi*a) * (R_ref/a)**(n-1),
    so
        I0 = -4*pi*a*B_ref/(mu0*N) * (a/R_ref)**(n-1).

    The MINUS SIGN is required: with I_i = +I0*cos(n.theta_i) and I0 > 0 the
    resulting field is B_y = -B_n (verified against thin-wire Biot-Savart to 6
    digits).  A positive B_ref therefore returns a negative I0.

    If `b` is given the winding is closed by returns at radius b carrying -I_i,
    which multiplies the body field by [1 - (a/b)**n]; I0 is divided by that
    factor to compensate.
    '''
    g = _resolve_geom(geom, geom_overrides)
    a, R_ref = g.a, g.R_ref
    I0 = -4 * np.pi * a * B_ref / (mu0 * N) * (a / R_ref) ** (n - 1)
    if b is not None:
        red = 1.0 - (a / b) ** n
        if abs(red) < 1e-9:
            raise ValueError('return radius b=%.4f gives no net field for n=%d' % (b, n))
        I0 = I0 / red
    return I0


def strength_for_current(n, I0, N, geom=None, b=None, **geom_overrides):
    '''Inverse of current_for_strength: B_n(R_ref) produced by amplitude I0.'''
    g = _resolve_geom(geom, geom_overrides)
    B = -mu0 * N * I0 / (4 * np.pi * g.a) * (g.R_ref / g.a) ** (n - 1)
    if b is not None:
        B = B * (1.0 - (g.a / b) ** n)
    return B


def bars_per_pole(n, N_target=32):
    '''Smallest multiple of 4n that is >= N_target.

    A multiple of 2n is required so that every bar has an exact partner carrying
    -I (theta_j = theta_i + pi/n must land on the bar grid), which is what makes
    the chord closure possible.  The extra factor of 2 additionally guarantees
    that no bar carries zero current.
    '''
    n = int(n)
    m = int(np.ceil(N_target / (4 * n)))
    return 4 * n * max(m, 1)


def bar_angles_grid(N):
    '''Bar azimuths theta_i = 2*pi*(i + 1/2)/N [rad], the half-offset grid.'''
    return 2 * np.pi * (np.arange(N) + 0.5) / N


def chord_segments(n, geom=None, **geom_overrides):
    '''Number of end-turn polyline segments k, and the vertex angles.

    The two bars of a pair are separated by d_theta = pi/n.  A single straight
    chord between them approaches the axis to a*cos(d_theta/2), which for n=1 is
    ZERO (the chord is a diameter straight through the bore) and for n=2 is only
    0.707*a.  Instead the connection is a k-segment polyline whose vertices lie
    on the cylinder r=a, so its closest approach is a*cos(d_theta/(2k)); k is the
    smallest integer with that >= r_clear.

    Returns (k, offsets) where `offsets` are the k+1 vertex angles relative to
    the first bar, in radians, spanning [0, d_theta].
    '''
    g = _resolve_geom(geom, geom_overrides)
    d_theta = np.pi / n
    k = 1
    while g.a * np.cos(d_theta / (2 * k)) < g.r_clear:
        k += 1
        if k > 1000:
            raise ValueError('cannot satisfy r_clear=%.4f with a=%.4f' % (g.r_clear, g.a))
    return k, np.linspace(0.0, d_theta, k + 1)


# ----------------------------------------------------------------------------
# closing a winding into current loops
# ----------------------------------------------------------------------------
def pair_bars(n, N):
    '''Perfect matching of bar indices into pairs carrying opposite currents.

    The partner of bar i is bar i+m with m = N/(2n), because
    theta_{i+m} = theta_i + pi/n and cos(n*theta) flips sign under that shift.
    But i -> i+m is not an involution, so it is not itself a matching: adding m
    repeatedly walks a cycle of length N/m = 2n.  Each such cycle is even, so we
    take alternate edges around it -- pairs (0,1), (2,3), ... of each cycle --
    which gives a perfect matching of all N bars into N/2 pairs.

    Returns a list of (i, j) index pairs with I[j] == -I[i].
    '''
    if N % (2 * n) != 0:
        raise ValueError('N=%d must be a multiple of 2n=%d for chord pairing'
                         % (N, 2 * n))
    m = N // (2 * n)
    pairs = []
    for r in range(m):                      # one cycle per residue class mod m
        cycle = [(r + j * m) % N for j in range(2 * n)]
        for j in range(0, 2 * n, 2):        # alternate edges of an even cycle
            pairs.append((cycle[j], cycle[j + 1]))
    return pairs


def _cyl(a, theta, z):
    return np.array([a * np.cos(theta), a * np.sin(theta), z])


def close_winding(theta, I, z0, z1, geom, n, closure='chord', b=None,
                  element='', skew=False, cond_N_start=0, name=''):
    '''Build the closed set of straight bars for one winding.

    theta : (N,) bar azimuths [rad]
    I     : (N,) signed bar currents [A]; for closure='chord' these must satisfy
            I[i+N/(2n)] == -I[i] (true for both cos(n.theta) and 2n-fold
            symmetric sector distributions).
    z0,z1 : axial extent of the straight (body) bars.

    closure='chord'  : each bar is paired with its opposite-current partner and
                       the two are joined at each end by a k-segment polyline
                       with vertices on r=a (k from chord_segments, so the end
                       turns never come inside geom.r_clear).
    closure='radial' : each bar returns on its own -- radially out to b, axially
                       back, radially in.  The body field is then reduced by
                       [1 - (a/b)**n].
    '''
    a = geom.a
    theta = np.asarray(theta, float)
    I = np.asarray(I, float)
    N = len(theta)
    rows = []
    cn = cond_N_start

    if closure == 'chord':
        k, offs = chord_segments(n, geom)
        for loop_id, (i, j) in enumerate(pair_bars(n, N)):
            if np.isclose(I[i], 0.0):
                continue
            if not np.isclose(I[j], -I[i], rtol=1e-9, atol=1e-9 * max(abs(I[i]), 1.0)):
                raise ValueError('bars %d and %d do not carry opposite currents '
                                 '(%.6g vs %.6g)' % (i, j, I[i], I[j]))
            # two body bars, each along +z with its own signed current
            for idx in (i, j):
                rows.append(make_bar_row(_cyl(a, theta[idx], z0),
                                         _cyl(a, theta[idx], z1),
                                         I[idx], cn, geom=geom,
                                         name='%s body bar %d' % (name, idx),
                                         psi_mode='radial', element=element,
                                         n=n, skew=skew, seg_kind='axial',
                                         loop=loop_id))
                cn += 1
            # end turns: at z1 current runs i -> j carrying +I[i];
            # at z0 it runs j -> i, i.e. the same i -> j segments with -I[i]
            for z, sgn, tag in ((z1, +1.0, 'end_hi'), (z0, -1.0, 'end_lo')):
                for s in range(k):
                    p0 = _cyl(a, theta[i] + offs[s], z)
                    p1 = _cyl(a, theta[i] + offs[s + 1], z)
                    rows.append(make_bar_row(p0, p1, sgn * I[i], cn, geom=geom,
                                             name='%s %s %d.%d' % (name, tag, i, s),
                                             psi_mode='normal', element=element,
                                             n=n, skew=skew, seg_kind=tag,
                                             loop=loop_id))
                    cn += 1

    elif closure == 'radial':
        if b is None:
            raise ValueError("closure='radial' needs a return radius b")
        if b <= a:
            raise ValueError('return radius b must exceed the winding radius a')
        for i in range(N):
            if np.isclose(I[i], 0.0):
                continue
            th = theta[i]
            pts = [(_cyl(a, th, z0), _cyl(a, th, z1), 'axial', 'radial'),
                   (_cyl(a, th, z1), _cyl(b, th, z1), 'radial_hi', 'normal'),
                   (_cyl(b, th, z1), _cyl(b, th, z0), 'return', 'radial'),
                   (_cyl(b, th, z0), _cyl(a, th, z0), 'radial_lo', 'normal')]
            for p0, p1, tag, pm in pts:
                rows.append(make_bar_row(p0, p1, I[i], cn, geom=geom,
                                         name='%s %s %d' % (name, tag, i),
                                         psi_mode=pm, element=element, n=n,
                                         skew=skew, seg_kind=tag, loop=i))
                cn += 1
    else:
        raise ValueError("closure must be 'chord' or 'radial'")
    return rows


def make_cosn_winding(n, L, zc, B_ref=None, I0=None, skew=False, geom=None,
                      closure='chord', b=None, N=None, cond_N_start=0,
                      element='', name=None, **geom_overrides):
    '''Closed cos(n.theta) (normal) or sin(n.theta) (skew) winding.

    Exactly one of B_ref [T] (the target B_n at geom.R_ref) or I0 [A] must be given.
    Returns a DataFrame of straight bars in the Mu2e schema.
    '''
    g = _resolve_geom(geom, geom_overrides)
    if N is None:
        N = bars_per_pole(n, g.N_target)
    if (B_ref is None) == (I0 is None):
        raise ValueError('give exactly one of B_ref or I0')
    if I0 is None:
        I0 = current_for_strength(n, B_ref, N, g, b=b)
    if name is None:
        name = element or ('%s%d-pole' % ('skew ' if skew else '', 2 * n))
    theta = bar_angles_grid(N)
    # skew = normal rotated by +90/n deg.  The extra minus makes a positive
    # B_ref give a POSITIVE A_n (skew dipole -> +B_x, the 'V' corrector),
    # matching the sign convention already fixed for the normal case.
    I = I0 * (-np.sin(n * theta) if skew else np.cos(n * theta))
    rows = close_winding(theta, I, zc - 0.5 * L, zc + 0.5 * L, g, n,
                         closure=closure, b=b, element=element, skew=skew,
                         cond_N_start=cond_N_start, name=name)
    return bars_to_df(rows)


# ----------------------------------------------------------------------------
# geometry checks (no field calculation)
# ----------------------------------------------------------------------------
def bar_endpoints(df):
    '''(N,3) start and end points of every bar, from the schema columns.'''
    p0 = df[['x0', 'y0', 'z0']].values.astype(float)
    if 'z1' in df.columns:
        z1 = df['z1'].values.astype(float)
    else:                                   # reconstruct from the Euler angles
        th = np.radians(df['theta2'].values.astype(float))
        z1 = df['z0'].values.astype(float) + df['length'].values.astype(float) * np.cos(th)
    p1 = np.column_stack([df['x1'].values.astype(float),
                          df['y1'].values.astype(float), z1])
    return p0, p1


def check_closure(df, tol=1e-9, decimals=9):
    '''Kirchhoff check: net current into every node must vanish.

    Returns (ok, worst_residual, df_nodes).  An open winding gives a field that
    is not divergence-free, so this must pass before any field calculation.
    '''
    p0, p1 = bar_endpoints(df)
    I = df['I'].values.astype(float)
    net = {}
    for pt, sgn in ((p0, -1.0), (p1, +1.0)):        # current leaves p0, enters p1
        for p, i in zip(pt, I):
            key = tuple(np.round(p, decimals))
            net[key] = net.get(key, 0.0) + sgn * i
    res = np.array(list(net.values()))
    scale = max(np.abs(I).max(), 1.0)
    worst = float(np.abs(res).max()) if len(res) else 0.0
    nodes = pd.DataFrame([list(k) + [v] for k, v in net.items()],
                         columns=['x', 'y', 'z', 'I_net'])
    return worst <= tol * scale, worst, nodes


def min_conductor_radius(df, n_sample=64):
    '''Closest approach of any conductor centreline to the z axis [m].'''
    p0, p1 = bar_endpoints(df)
    t = np.linspace(0.0, 1.0, n_sample)[None, :, None]
    pts = p0[:, None, :] + t * (p1 - p0)[:, None, :]
    return float(np.hypot(pts[..., 0], pts[..., 1]).min())


def integration_nodes(df, dxyz):
    '''Node counts (nx, ny, nz) helicalc will use for each bar.

    helicalc builds linspace(-W/2, W/2, int(W/dxyz[0] + 1)) and so on.  If that
    count is 1 the trapezoid integral over the dimension is silently ZERO, so
    this is checked before committing GPU time.
    '''
    W = df['W'].values.astype(float)
    T = df['T'].values.astype(float)
    L = df['length'].values.astype(float)
    nx = np.abs((W / dxyz[0] + 1).astype(int))
    ny = np.abs((T / dxyz[1] + 1).astype(int))
    nz = np.abs((L / dxyz[2] + 1).astype(int))
    return nx, ny, nz


# ----------------------------------------------------------------------------
# thin-wire reference field (numpy only, no GPU)
# ----------------------------------------------------------------------------
def thin_wire_field(df, points):
    '''Exact Biot-Savart for finite straight segments on the bar centrelines.

    points : (M,3) array [m].  Returns (M,3) B in tesla.

    This ignores the conductor cross-section, so it differs from the full 3D
    integral at O((T/d)^2) for a field point a distance d from the bar, but it
    needs no GPU and is exact in the thin-wire limit -- ideal as a first-line
    regression on signs, orientations and the cos(n.theta) normalisation.
    '''
    pts = np.atleast_2d(np.asarray(points, float))
    p0, p1 = bar_endpoints(df)
    I = df['I'].values.astype(float)
    B = np.zeros_like(pts)
    for A, C, cur in zip(p0, p1, I):
        if cur == 0.0:
            continue
        # standard finite-segment formula
        a_vec = pts - A            # (M,3)
        b_vec = pts - C
        seg = C - A
        Lseg = np.linalg.norm(seg)
        if Lseg == 0.0:
            continue
        ehat = seg / Lseg
        cross = np.cross(ehat, a_vec)          # (M,3), = rho_hat * rho
        rho2 = np.einsum('ij,ij->i', cross, cross)
        with np.errstate(divide='ignore', invalid='ignore'):
            na = np.linalg.norm(a_vec, axis=1)
            nb = np.linalg.norm(b_vec, axis=1)
            cos_a = np.einsum('ij,j->i', a_vec, ehat) / na
            cos_b = np.einsum('ij,j->i', b_vec, ehat) / nb
            fac = mu0 * cur / (4 * np.pi) * (cos_a - cos_b) / rho2
        fac = np.where(rho2 > 1e-24, fac, 0.0)
        B += fac[:, None] * cross
    return B


def harmonics_from_field(Bx, By, phi, n_max=20):
    '''Multipole coefficients from B sampled on a circle.

    Bx, By : field components at azimuths `phi` (uniformly spaced, radius R).
    Returns a dict {order: (B_n, A_n)} where, following the paper convention,
        B_y + i B_x = sum_n (B_n + i A_n) * exp(i (n-1) phi)   at r = R.
    '''
    M = len(phi)
    f = By + 1j * Bx
    C = np.fft.fft(f) / M                      # C[k] multiplies exp(i k phi)
    out = {}
    for order in range(1, n_max + 1):
        c = C[order - 1]
        out[order] = (c.real, c.imag)
    return out


# ----------------------------------------------------------------------------
# sector-block windings (realistic allowed harmonics)
# ----------------------------------------------------------------------------
def sector_half_width(n):
    '''Maximum block half-width for a 2n-pole [deg].

    Blocks are measured as an angular distance from a pole centre; the sign of
    cos(n.theta) flips at pi/(2n), so a block may not extend past 90/n degrees.
    '''
    return 90.0 / n


def _sector_S(m, blocks):
    '''Thin-shell harmonic integral sum(sin(m*hi) - sin(m*lo))/m over blocks [rad].'''
    tot = 0.0
    for lo, hi in blocks:
        tot += (np.sin(m * hi) - np.sin(m * lo)) / m
    return tot


def sector_harmonics(n, blocks, geom=None, orders=(6, 10, 14), **geom_overrides):
    '''Allowed harmonics b_m/b_n in units (1e-4 of the main field) at geom.R_ref.

    blocks : list of (lo, hi) angular block edges in DEGREES, measured from a
             pole centre, each within [0, sector_half_width(n)].  Blocks are
             mirrored about the pole and repeated with 2n-fold alternating sign.

    Thin-shell result:
        b_m/b_n = 1e4 * [S_m / S_n] * (R_ref/a)**(m-n),
        S_m = sum_blocks [sin(m*hi) - sin(m*lo)] / m
    Only m = n, 3n, 5n, ... are allowed; others vanish identically.
    '''
    g = _resolve_geom(geom, geom_overrides)
    br = [(np.radians(lo), np.radians(hi)) for lo, hi in blocks]
    Sn = _sector_S(n, br)
    if abs(Sn) < 1e-12:
        raise ValueError('blocks produce no main field')
    return {m: 1e4 * _sector_S(m, br) / Sn * (g.R_ref / g.a) ** (m - n)
            for m in orders}


def solve_sector_blocks(n, targets, geom=None, n_blocks=2, x0=None,
                        coverage_deg=None, **geom_overrides):
    '''Solve block edges giving target allowed harmonics.

    targets : {order: value_in_units}, e.g. {6: +1.0, 10: -0.5} for a quad.
    n_blocks: number of angular blocks per pole (2 = one block plus a wedge).
    coverage_deg : optional target for the total energised angle, used to pick
              among the solution family when there are more edges than targets.

    Returns a list of (lo, hi) block edges in degrees, sorted and within
    [0, sector_half_width(n)].  Re-solve whenever the aperture, a or R_ref
    changes -- the (R_ref/a)**(m-n) scaling makes the answer geometry-specific.
    '''
    from scipy.optimize import least_squares
    g = _resolve_geom(geom, geom_overrides)
    half = sector_half_width(n)
    orders = sorted(targets)
    # edges: block 0 starts at 0, so the free parameters are
    # [hi_0, lo_1, hi_1, lo_2, hi_2, ...] -> 2*n_blocks - 1 of them
    n_free = 2 * n_blocks - 1

    def to_blocks(p):
        e = np.concatenate([[0.0], np.asarray(p, float)])
        return [(e[2 * i], e[2 * i + 1]) for i in range(n_blocks)]

    def resid(p):
        blocks = to_blocks(p)
        h = sector_harmonics(n, blocks, g, orders=orders)
        r = [h[m] - targets[m] for m in orders]
        if coverage_deg is not None:
            r.append(sum(hi - lo for lo, hi in blocks) - coverage_deg)
        # keep the edges ordered
        e = np.concatenate([[0.0], np.asarray(p, float)])
        r.extend(100.0 * np.minimum(np.diff(e), 0.0))
        return r

    if x0 is None:
        x0 = np.linspace(0.45 * half, 0.96 * half, n_free)
    best = None
    for scale in (1.0, 0.8, 0.6, 1.15):
        seed = np.clip(np.asarray(x0, float) * scale, 1e-3, half)
        seed = np.sort(seed)
        try:
            r = least_squares(resid, seed, bounds=(1e-6, half))
        except Exception:
            continue
        e = np.concatenate([[0.0], r.x])
        if np.all(np.diff(e) > 0) and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        raise RuntimeError('no admissible sector-block solution found')
    return [(round(lo, 6), round(hi, 6)) for lo, hi in to_blocks(best.x)]


def sector_bar_currents(n, theta, blocks, I0, edge_mode='fractional'):
    """Signed currents for a sector winding on the bar grid `theta` [rad].

    A bar's angular distance to the nearest pole centre (k*pi/n) decides whether
    it is energised; its sign is sign(cos(n*theta)), which makes the pattern
    2n-fold antisymmetric and hence exactly compatible with the chord pairing
    (partners always carry +/- the same magnitude).

    edge_mode:
      'fractional' (default) -- each bar owns an angular cell of width 2*pi/N and
            carries I0 times the fraction of that cell lying inside a block.
            Interior bars get exactly I0; only the one bar straddling each block
            edge is partial (physically: a partial turn or graded cable there).
      'binary' -- strictly uniform |I0|, bar in or out by its centre.

    'binary' is the naive choice but makes the harmonics a STEP function of the
    block edges, quantised by the bar pitch: near a quad's working point
    db6/d(edge) ~ 37 units/deg, so an N=32 winding (11.25 deg pitch) quantises b6
    at the ~400-unit level and ~14000 bars would be needed for unit-level
    control.  'fractional' restores continuous, tunable dependence on the edges
    and converges to the thin-shell formula.
    """
    theta = np.asarray(theta, float)
    N = len(theta)
    pole_pitch = np.pi / n
    d = np.abs((theta + 0.5 * pole_pitch) % pole_pitch - 0.5 * pole_pitch)
    d_deg = np.degrees(d)
    sign = np.sign(np.cos(n * theta))

    if edge_mode == 'binary':
        w = np.zeros(N)
        for lo, hi in blocks:
            w[(d_deg >= lo - 1e-9) & (d_deg <= hi + 1e-9)] = 1.0
    elif edge_mode == 'fractional':
        half_cell = 0.5 * (360.0 / N)          # half the bar angular pitch [deg]
        w = np.zeros(N)
        for lo, hi in blocks:
            # overlap of each bar's cell [d-half, d+half] with the block [lo, hi],
            # folded about d=0 so a block touching the pole centre is not clipped
            a_lo = d_deg - half_cell
            a_hi = d_deg + half_cell
            ov = np.minimum(a_hi, hi) - np.maximum(a_lo, lo)
            ov_mirror = np.minimum(-a_lo, hi) - np.maximum(-a_hi, lo)
            w += (np.clip(ov, 0.0, None) + np.clip(ov_mirror, 0.0, None)) / (2 * half_cell)
        w = np.clip(w, 0.0, 1.0)
    else:
        raise ValueError("edge_mode must be 'fractional' or 'binary'")
    return sign * I0 * w


def make_sector_winding(n, L, zc, blocks, B_ref=None, I0=None, skew=False,
                        geom=None, closure='chord', b=None, N=None,
                        cond_N_start=0, element='', name=None,
                        edge_mode='fractional', **geom_overrides):
    '''Closed sector-block winding: uniform |I| bars in angular blocks.

    Unlike a cos(n.theta) shell this has realistic allowed harmonics b_{n(2j+1)}
    set by the block edges (see solve_sector_blocks).  All bars carry the same
    |I0|, so the chord pairing is exact by construction.

    If B_ref is given, I0 is scaled so the thin-shell main field matches it.
    '''
    g = _resolve_geom(geom, geom_overrides)
    if N is None:
        N = bars_per_pole(n, g.N_target)
    if (B_ref is None) == (I0 is None):
        raise ValueError('give exactly one of B_ref or I0')
    half = sector_half_width(n)
    for lo, hi in blocks:
        if lo < -1e-9 or hi > half + 1e-9 or hi <= lo:
            raise ValueError('block (%.4f, %.4f) deg outside [0, %.4f] for n=%d'
                             % (lo, hi, half, n))
    theta = bar_angles_grid(N)
    if skew:
        theta_eff = theta - np.pi / (2 * n)     # rotate the pattern by 90/n deg
    else:
        theta_eff = theta

    if I0 is None:
        # scale from the equivalent cos(n.theta) shell: a sector distribution
        # with the same peak current has main-field weight sum|cos| -> S_n
        probe = sector_bar_currents(n, theta_eff, blocks, 1.0, edge_mode)
        w = np.sum(probe * np.cos(n * theta_eff))
        w_cos = 0.5 * N                          # sum cos^2(n.theta) over the grid
        I0 = current_for_strength(n, B_ref, N, g, b=b) * w_cos / w
    I = sector_bar_currents(n, theta_eff, blocks, I0, edge_mode)
    if skew:
        I = -I
    if name is None:
        name = element or ('%ssector %d-pole' % ('skew ' if skew else '', 2 * n))
    rows = close_winding(theta, I, zc - 0.5 * L, zc + 0.5 * L, g, n,
                         closure=closure, b=b, element=element, skew=skew,
                         cond_N_start=cond_N_start, name=name)
    return bars_to_df(rows)


# ----------------------------------------------------------------------------
# analytic 2D harmonics of a discrete bar shell
# ----------------------------------------------------------------------------
def multipole_coeffs_2d(theta, I, geom=None, orders=range(1, 21), **geom_overrides):
    '''Exact interior multipole coefficients of infinite line currents at r=a.

    For wires at (a, theta_i) carrying I_i, the interior field is
        B_y + i B_x = sum_m (B_m + i A_m) ((x+iy)/R_ref)**(m-1)
    with
        B_m = -(mu0 / (2 pi a)) * (R_ref/a)**(m-1) * sum_i I_i cos(m theta_i)
        A_m = -(mu0 / (2 pi a)) * (R_ref/a)**(m-1) * sum_i I_i sin(m theta_i)

    This is the 2D body limit (no end turns, no cross-section), and it is what
    the sector-block solver targets.  Returns {m: (B_m, A_m)} in tesla.
    '''
    g = _resolve_geom(geom, geom_overrides)
    theta = np.asarray(theta, float)
    I = np.asarray(I, float)
    pre = -mu0 / (2 * np.pi * g.a)
    out = {}
    for m in orders:
        scale = pre * (g.R_ref / g.a) ** (m - 1)
        out[m] = (scale * np.sum(I * np.cos(m * theta)),
                  scale * np.sum(I * np.sin(m * theta)))
    return out


def discrete_sector_harmonics(n, blocks, N, geom=None, orders=(6, 10, 14),
                              edge_mode='fractional', **geom_overrides):
    '''Allowed harmonics in units for the DISCRETE sector winding of N bars.

    Unlike sector_harmonics (a continuous thin shell) this accounts for the
    finite bar count, which matters: at N=32 the discretisation shifts a quad's
    b6 by tens of units.  Use this, not the thin-shell version, to pick block
    edges for a winding that will actually be built.
    '''
    g = _resolve_geom(geom, geom_overrides)
    theta = bar_angles_grid(N)
    I = sector_bar_currents(n, theta, blocks, 1.0, edge_mode)
    c = multipole_coeffs_2d(theta, I, g, orders=tuple(orders) + (n,))
    main = c[n][0]
    if abs(main) < 1e-30:
        raise ValueError('blocks produce no main field')
    return {m: 1e4 * c[m][0] / main for m in orders}


def solve_sector_blocks_discrete(n, targets, N, geom=None, n_blocks=2, x0=None,
                                 edge_mode='fractional', **geom_overrides):
    '''Solve block edges so the DISCRETE N-bar winding hits the target harmonics.

    Same interface as solve_sector_blocks but exact for the winding that gets
    built.  Requires edge_mode='fractional' for a continuously tunable solution
    (with 'binary' the harmonics are a step function of the edges).
    '''
    from scipy.optimize import least_squares
    g = _resolve_geom(geom, geom_overrides)
    half = sector_half_width(n)
    orders = sorted(targets)
    n_free = 2 * n_blocks - 1

    def to_blocks(p):
        e = np.concatenate([[0.0], np.asarray(p, float)])
        return [(e[2 * i], e[2 * i + 1]) for i in range(n_blocks)]

    def resid(p):
        blocks = to_blocks(p)
        try:
            h = discrete_sector_harmonics(n, blocks, N, g, orders, edge_mode)
        except ValueError:
            return [1e6] * (len(orders) + n_free)
        r = [h[m] - targets[m] for m in orders]
        e = np.concatenate([[0.0], np.asarray(p, float)])
        r.extend(100.0 * np.minimum(np.diff(e), 0.0))
        return r

    seeds = []
    if x0 is not None:
        seeds.append(np.asarray(x0, float))
    try:
        seeds.append(np.array([e for blk in solve_sector_blocks(
            n, targets, g, n_blocks=n_blocks) for e in blk][1:]))
    except Exception:
        pass
    seeds.append(np.linspace(0.45 * half, 0.96 * half, n_free))
    seeds.append(np.linspace(0.30 * half, 0.90 * half, n_free))

    best = None
    for seed in seeds:
        seed = np.sort(np.clip(np.asarray(seed, float), 1e-3, half))
        try:
            r = least_squares(resid, seed, bounds=(1e-6, half))
        except Exception:
            continue
        e = np.concatenate([[0.0], r.x])
        if np.all(np.diff(e) > 0) and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        raise RuntimeError('no admissible discrete sector-block solution found')
    return [(round(lo, 6), round(hi, 6)) for lo, hi in to_blocks(best.x)]


# ----------------------------------------------------------------------------
# localized perturbation
# ----------------------------------------------------------------------------
def make_square_loop(center, moment_dir, moment, side=0.04, W=None, T=None,
                     geom=None, cond_N_start=0, element='PERT', name='pert'):
    '''Small square current loop with a prescribed magnetic moment [A m^2].

    Stands in for the handoff's point dipole (m = 15 A m^2 along y, the induced
    moment of a ~20 mm steel sphere in the quad fringe), but built from four
    straight bars so the whole truth field stays Biot-Savart from conductors.
    Current is set from I = moment / side**2.
    '''
    g = _resolve_geom(geom, {})
    if geom is not None:
        g = geom
    W = g.W / 4 if W is None else W
    T = g.T / 4 if T is None else T
    c = np.asarray(center, float)
    nhat = np.asarray(moment_dir, float)
    nhat = nhat / np.linalg.norm(nhat)
    # two unit vectors spanning the loop plane; current circulates right-handed
    # about nhat so the moment points along nhat
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, nhat)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    u = np.cross(nhat, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(nhat, u)
    h = 0.5 * side
    corners = [c - h * u - h * v, c + h * u - h * v,
               c + h * u + h * v, c - h * u + h * v]
    I = moment / side ** 2
    rows = []
    for i in range(4):
        rows.append(make_bar_row(corners[i], corners[(i + 1) % 4], I,
                                 cond_N_start + i, geom=g, W=W, T=T,
                                 name='%s side %d' % (name, i), psi_mode='normal',
                                 element=element, n=0, skew=False,
                                 seg_kind='pert', loop=0))
    return bars_to_df(rows)


# ----------------------------------------------------------------------------
# the HL-LHC-like assembly
# ----------------------------------------------------------------------------
# Handoff section 2.  z positions are effective-field-boundary positions in the
# toy frame (real HL-LHC v1.5 positions, z_toy = z_real - 64.916 m, with the Q3b
# body and the Q3b->MCBXFA drift compressed).  Mapping volume: r <= R_map,
# 0 <= z <= 8.6 m.  Strengths are HL-LHC maxima.
#
# Keep this table as the single source of truth: the builder, the plotting
# scripts and the validation scripts all read it, so figures and generated
# geometry cannot drift apart.
ASSEMBLY_ELEMENTS = [
    # name,  real,      n, skew,  B_n(R_ref)[T], z0,    z1,     kind
    dict(name='MQ',   real='Q3b (MQXFA) end', n=2, skew=False, B_ref=6.60,
         z0=0.400, z1=1.900, kind='main', corrector=False),
    dict(name='MCBH', real='MCBXFA, H plane', n=1, skew=False, B_ref=2.05,
         z0=2.400, z1=4.600, kind='dipole', corrector=True, BdL=4.5),
    dict(name='MCBV', real='MCBXFA, V plane', n=1, skew=True,  B_ref=2.05,
         z0=2.400, z1=4.600, kind='dipole', corrector=True, BdL=4.5,
         a_offset=0.010),     # nested coil: 10 mm outside the H winding
    dict(name='MQS',  real='MQSXF',  n=2, skew=True,  B_ref=1.74,
         z0=4.913, z1=5.315, kind='corr', corrector=True, BdL=0.700),
    dict(name='MCT',  real='MCTXF',  n=6, skew=False, B_ref=0.18,
         z0=5.520, z1=5.989, kind='corr', corrector=True, BdL=0.086),
    dict(name='MCTS', real='MCTSXF', n=6, skew=True,  B_ref=0.17,
         z0=6.135, z1=6.234, kind='corr', corrector=True, BdL=0.017),
    dict(name='MCD',  real='MCDXF',  n=5, skew=False, B_ref=0.26,
         z0=6.392, z1=6.537, kind='corr', corrector=True, BdL=0.037),
    dict(name='MCDS', real='MCDSXF', n=5, skew=True,  B_ref=0.26,
         z0=6.682, z1=6.827, kind='corr', corrector=True, BdL=0.037),
    dict(name='MCO',  real='MCOXF',  n=4, skew=False, B_ref=0.48,
         z0=6.972, z1=7.117, kind='corr', corrector=True, BdL=0.069),
    dict(name='MCOS', real='MCOSXF', n=4, skew=True,  B_ref=0.48,
         z0=7.262, z1=7.407, kind='corr', corrector=True, BdL=0.069),
    dict(name='MCS',  real='MCSXF',  n=3, skew=False, B_ref=0.57,
         z0=7.540, z1=7.708, kind='corr', corrector=True, BdL=0.095),
    dict(name='MCSS', real='MCSSXF', n=3, skew=True,  B_ref=0.57,
         z0=7.850, z1=8.018, kind='corr', corrector=True, BdL=0.095),
]

# localized perturbation (handoff section 2): ~9 G on axis
PERT_SPEC = dict(name='PERT', center=(0.12, 0.0, 2.15), moment_dir=(0.0, 1.0, 0.0),
                 moment=15.0, side=0.04)

MAPPING_VOLUME = dict(z0=0.0, z1=8.6)

# quad allowed-harmonic targets, in units at R_ref (handoff section 2)
MQ_HARMONIC_TARGETS = {6: +1.0, 10: -0.5}


def build_assembly(layout='full', corrector_scale=1.0, quad_harmonics='sector',
                   geom=None, closure='chord', b=None, include_pert=True,
                   mq_targets=None, edge_mode='fractional', verbose=False,
                   **geom_overrides):
    '''Build the full HL-LHC-like multipole assembly as one bar DataFrame.

    layout : 'full' (all 13 windings) or 'quad' (MQ only) or a list of names.
    corrector_scale : multiplies every corrector strength (1.0 = HL-LHC maxima).
    quad_harmonics :
        'sector'   -- MQ built from sector blocks solved to hit mq_targets
                      exactly for the DISCRETE winding (handoff 3.4a, default)
        'explicit' -- MQ is a pure cos(2.theta) shell plus weak n=6 and n=10
                      windings injecting the same harmonics (handoff 3.4b)
        None       -- MQ is a pure cos(2.theta) shell, no allowed harmonics
    geom / geom_overrides : MultipoleGeom; pass aperture=... to rescale the
        whole machine consistently.

    Returns a DataFrame with unique numeric `cond N`, ready for
    StraightIntegrator3D, plus bookkeeping columns (element, n, skew, seg_kind).
    '''
    g = _resolve_geom(geom, geom_overrides)
    if mq_targets is None:
        mq_targets = dict(MQ_HARMONIC_TARGETS)
    if layout == 'full':
        names = [e['name'] for e in ASSEMBLY_ELEMENTS]
    elif layout == 'quad':
        names = ['MQ']
    else:
        names = list(layout)

    parts = []
    cn = 0
    for spec in ASSEMBLY_ELEMENTS:
        if spec['name'] not in names:
            continue
        n = spec['n']
        L = spec['z1'] - spec['z0']
        zc = 0.5 * (spec['z0'] + spec['z1'])
        B = spec['B_ref'] * (corrector_scale if spec.get('corrector') else 1.0)
        # nested dipole: second plane sits at a slightly larger radius
        g_el = g.with_(a=g.a + spec['a_offset']) if 'a_offset' in spec else g
        N = bars_per_pole(n, g_el.N_target)

        if spec['name'] == 'MQ' and quad_harmonics == 'sector':
            blocks = solve_sector_blocks_discrete(n, mq_targets, N, g_el,
                                                  edge_mode=edge_mode)
            df = make_sector_winding(n, L, zc, blocks, B_ref=B, skew=spec['skew'],
                                     geom=g_el, closure=closure, b=b, N=N,
                                     cond_N_start=cn, element=spec['name'],
                                     edge_mode=edge_mode)
            if verbose:
                print('  %s sector blocks (deg): %s' % (spec['name'], blocks))
        else:
            df = make_cosn_winding(n, L, zc, B_ref=B, skew=spec['skew'],
                                   geom=g_el, closure=closure, b=b, N=N,
                                   cond_N_start=cn, element=spec['name'])
        parts.append(df)
        cn += len(df)

        if spec['name'] == 'MQ' and quad_harmonics == 'explicit':
            # weak explicit high-order windings carrying the target harmonics
            for m, units in sorted(mq_targets.items()):
                B_m = units * 1e-4 * B
                N_m = bars_per_pole(m, g_el.N_target)
                df_m = make_cosn_winding(m, L, zc, B_ref=B_m, skew=False,
                                         geom=g_el, closure=closure, b=b, N=N_m,
                                         cond_N_start=cn,
                                         element='%s_b%d' % (spec['name'], m))
                parts.append(df_m)
                cn += len(df_m)

    if include_pert and (layout == 'full' or 'PERT' in names):
        parts.append(make_square_loop(PERT_SPEC['center'], PERT_SPEC['moment_dir'],
                                      PERT_SPEC['moment'], PERT_SPEC['side'],
                                      geom=g, cond_N_start=cn,
                                      element=PERT_SPEC['name']))

    df = pd.concat(parts, ignore_index=True)
    df['cond N'] = np.arange(len(df), dtype=int)     # unique and numeric
    return df


# ----------------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------------
def assembly_param_path(name):
    from helicalc import helicalc_dir
    import os
    return os.path.join(helicalc_dir, 'dev', 'params',
                        '%s_Straight_Bars.csv' % name)


def save_assembly_csv(df, name='Multipole_HLLHC_V1', path=None):
    '''Write the bar DataFrame to dev/params/<name>_Straight_Bars.csv.'''
    path = assembly_param_path(name) if path is None else path
    df.to_csv(path, index=False)
    return path


def load_assembly_csv(name='Multipole_HLLHC_V1', path=None):
    '''Read back an assembly written by save_assembly_csv.'''
    path = assembly_param_path(name) if path is None else path
    return pd.read_csv(path)


def summarize_assembly(df, geom=None):
    '''Per-element summary table: bars, conductor length, peak current.'''
    grp = df.groupby('element', sort=False)
    out = pd.DataFrame({
        'bars': grp['cond N'].size(),
        'loops': grp['loop'].nunique(),
        'n': grp['n'].first(),
        'skew': grp['skew'].first(),
        'cond_length_m': grp['length'].sum(),
        'I_peak_A': grp['I'].apply(lambda s: np.abs(s).max()),
    })
    return out.reset_index()
