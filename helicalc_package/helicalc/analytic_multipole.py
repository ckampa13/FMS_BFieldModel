'''
Analytic, exactly Maxwellian multipole fields (no conductor discretisation).

Both models write the scalar potential of one 2n-pole as

    psi = -Q(r, z) * sin(n phi)   (normal)      B = -grad psi  [tesla]
    psi = -Q(r, z) * cos(n phi)   (skew)

so that in the body B_y + i B_x = (B_n + i A_n) ((x + i y) / R_ref)^(n-1)
(the LHC convention; helicalc's MCBH gives B_y = +B_ref and MCBV B_x = +B_ref).
A model supplies Q, dQ/dr, dQ/dz and (n/r) Q.

Model A -- generalised gradients (Venturini-Dragt):
    Q = sum_m gamma_m C^(2m)(z) r^(n+2m),   gamma_m = (-1)^m n! (2m)! / (4^m m! (n+m)!) / (2m)!
  i.e. Q = sum_m (-1)^m n! / (4^m m! (n+m)!) C^(2m) r^(n+2m).  C(z) = B_n / (n R_ref^(n-1)) * S(z),
  S = E((z - z1)/D) E((z0 - z)/D), E(s) = 1 / (1 + exp(a0 + a1 s + ... + a5 s^5)).
  Derivatives of S come from exact truncated-Taylor-series arithmetic, so any order is cheap
  and exact to round-off.  The series converges only for r below the distance from z to the
  nearest complex pole of S; `series_terms` reports the size of every term so the truncation
  can be checked.

Model B -- ideal cos(n theta) current sheet on r = a over [z0, z1], closed by azimuthal
  current in the end planes (K_phi from div K = 0).  With u = z - z0, v = z - z1,
    Q = pref * int_0^inf K_n'(k a) I_n(k r) [sin(k u) - sin(k v)] dk,
    pref = -2 B_n (a / R_ref)^(n-1) a^2 / (n pi)       (mu0 K0 = 2 B_n (a/R_ref)^(n-1)).
  Evaluated by the midpoint rule in k.  Every node is an exact solution of Laplace's equation,
  so div B and curl B vanish to round-off whatever the step; the step only sets how closely
  the sum approximates the integral (periodic images at 4 pi / dk in z, cut-off e^(-k (a - r))).
  Realistic ends: the reversed 1/d^2 dipole tail beyond each end comes out automatically.

numpy/scipy only (no torch).  Runs in the old helicalc env (py3.8, scipy 1.5).
'''
import numpy as np
from scipy.special import ive, kve

MU0 = 4e-7 * np.pi


# ---------------------------------------------------------------------------
# truncated Taylor-series arithmetic (coefficients in powers of h, per point)
# ---------------------------------------------------------------------------
def _ts_mul(a, b):
    '''product of two coefficient arrays (..., M) -> (..., M)'''
    M = a.shape[-1]
    out = np.zeros(np.broadcast(a, b).shape)
    for k in range(M):
        out[..., k] = np.sum(a[..., :k + 1] * b[..., k::-1], axis=-1)
    return out


def _ts_exp_shifted(g):
    '''F with exp(g) = exp(g_0) * F, F_0 = 1 (keeps the recurrence in range)'''
    M = g.shape[-1]
    F = np.zeros_like(g)
    F[..., 0] = 1.0
    j = np.arange(M)
    for k in range(1, M):
        F[..., k] = np.sum(j[1:k + 1] * g[..., 1:k + 1] * F[..., k - 1::-1][..., :k], axis=-1) / k
    return F


def _ts_inv(w):
    '''1 / w'''
    M = w.shape[-1]
    q = np.zeros_like(w)
    q[..., 0] = 1.0 / w[..., 0]
    for k in range(1, M):
        q[..., k] = -np.sum(w[..., 1:k + 1] * q[..., k - 1::-1][..., :k], axis=-1) / w[..., 0]
    return q


def enge_series(s0, ds_dh, coeffs, M):
    '''Taylor coefficients in h of E(s0 + ds_dh * h), E(s) = 1 / (1 + exp(sum_i c_i s^i)).

    s0: array of expansion points; ds_dh: scalar; returns (len(s0), M).
    E = sigma(-p) obeys dE/dh = -(dp/dh) E (1 - E), which gives the coefficients by recurrence
    with every quantity bounded -- no exp of a large p (an exp/reciprocal recurrence overflows
    where |p| ~ 50 and its coefficients grow like (p'/D)^k / k!).
    '''
    s0 = np.atleast_1d(np.asarray(s0, float))
    # p(s0 + d h) as a polynomial in h (exact, degree len(coeffs)-1)
    lin = np.zeros((len(s0), M))
    lin[:, 0] = s0
    if M > 1:
        lin[:, 1] = ds_dh
    p = np.zeros((len(s0), M))
    powk = np.zeros((len(s0), M))
    powk[:, 0] = 1.0
    for c in coeffs:
        p += c * powk
        powk = _ts_mul(powk, lin)
    dp = np.zeros_like(p)                       # dp/dh
    dp[:, :-1] = p[:, 1:] * np.arange(1, M)
    q = np.zeros_like(p)
    u = np.zeros_like(p)                        # u = q (1 - q)
    # q_0 = sigma(-p_0), 1 - q_0 = sigma(p_0), both evaluated without overflow
    q[:, 0] = 0.5 * (1.0 - np.tanh(0.5 * p[:, 0]))
    e = np.exp(-np.abs(p[:, 0]))
    u[:, 0] = e / (1.0 + e) ** 2
    for k in range(M - 1):
        # (k+1) q_(k+1) = -sum_j dp_j u_(k-j)
        q[:, k + 1] = -np.sum(dp[:, :k + 1] * u[:, k::-1], axis=1) / (k + 1)
        m = k + 1
        # u_m = q_m (1 - 2 q_0) - sum_(i=1..m-1) q_i q_(m-i)
        u[:, m] = q[:, m] * (1.0 - 2.0 * q[:, 0]) - np.sum(q[:, 1:m] * q[:, m - 1:0:-1], axis=1)
    return q


def calibration_scale(elem):
    '''B_ref / (n-th harmonic of B_r at R_ref, at the magnet centre).

    B_r(R_ref, phi, z_c) = dQ/dr(R_ref, z_c) {sin, cos}(n phi), so its harmonic amplitude is dQ/dr.
    '''
    zc = 0.5 * (elem.z0 + elem.z1)
    dQr = elem.Q(elem.R_ref, np.array([zc]))[1][0]
    return elem.B_ref / dQr


# ---------------------------------------------------------------------------
# model A: generalised gradients with Enge ends
# ---------------------------------------------------------------------------
class GGMultipole(object):
    '''One 2n-pole from generalised gradients, C(z) = B_n / (n R_ref^(n-1)) S(z).

    n, skew, B_ref [T at R_ref], R_ref [m], z0, z1 [m] (the edges of S), D [m] (Enge length),
    enge = (a0, a1, ..., a5), shared by both ends (the same polynomial in the outward distance).
    '''

    def __init__(self, n, skew, B_ref, z0, z1, R_ref=0.05, D=0.15, enge=(0.0, 4.0), name='',
                 calibrate=True):
        self.n, self.skew, self.B_ref = int(n), bool(skew), float(B_ref)
        self.R_ref, self.z0, self.z1, self.D = float(R_ref), float(z0), float(z1), float(D)
        self.enge = tuple(float(c) for c in enge)
        self.name = name
        self.C0 = self.B_ref / (self.n * self.R_ref ** (self.n - 1))
        self.scale = 1.0
        if calibrate:
            self.scale = calibration_scale(self)
            self.C0 *= self.scale

    def S_taylor(self, z, M):
        '''Taylor coefficients of S about each z, in powers of (h [m])'''
        z = np.atleast_1d(np.asarray(z, float))
        e1 = enge_series((z - self.z1) / self.D, 1.0 / self.D, self.enge, M)
        e2 = enge_series((self.z0 - z) / self.D, -1.0 / self.D, self.enge, M)
        return _ts_mul(e1, e2)

    def gamma(self, mmax):
        '''gamma_m (2m)!, i.e. the coefficient of t_2m r^(n+2m) in Q, for m = 0..mmax'''
        g = np.empty(mmax + 1)
        g[0] = 1.0
        for m in range(mmax):
            g[m + 1] = -g[m] * (2 * m + 1) * (2 * m + 2) / (4.0 * (m + 1) * (self.n + m + 1))
        return g

    def series_terms(self, r, z, mmax=30):
        '''per-term contributions to (Q, dQ/dr, dQ/dz, nQ/r): arrays (len(z), mmax+1) each (r scalar)'''
        M = 2 * mmax + 2
        t = self.C0 * self.S_taylor(z, M)                # C^(j)/j!
        g = self.gamma(mmax)
        m = np.arange(mmax + 1)
        n = self.n
        rp = r ** (n + 2 * m)
        rpm1 = float(r) ** (n + 2 * m - 1)
        t2m = t[:, 2 * m]
        # d/dz of C^(2m) = C^(2m+1) = (2m+1)! t_(2m+1); in units of t: (2m+1) t_(2m+1) / t_(2m) scaling via gamma
        t2m1 = t[:, 2 * m + 1] * (2 * m + 1)
        Q = g * t2m * rp
        dQr = g * (n + 2 * m) * t2m * rpm1
        dQz = g * t2m1 * rp
        nQr = g * n * t2m * rpm1
        return Q, dQr, dQz, nQr

    def Q(self, r, z, mmax=30):
        '''(Q, dQ/dr, dQ/dz, nQ/r) summed to mmax, for scalar r and array z'''
        return tuple(a.sum(axis=1) for a in self.series_terms(r, z, mmax))


# ---------------------------------------------------------------------------
# model B: ideal cos(n theta) current sheet
# ---------------------------------------------------------------------------
class SheetMultipole(object):
    '''Ideal cos(n theta) surface current on r = a over [z0, z1] with end closure.

    The body field is B_n at R_ref (normal) or A_n (skew).  dk, kcut: midpoint-rule step and
    the cut-off exponent (k_max = kcut / (a - r_max)).
    calibrate: scale the current so the n-th harmonic at R_ref at the magnet centre is exactly
    B_ref (as helicalc's calibrate_winding does).  Without it the sheet has the 2D current, and a
    finite sheet then sits ~0.3 % high in the body: int B dz = B_ref L exactly, and the
    reversed tails outside take their share.
    '''

    def __init__(self, n, skew, B_ref, z0, z1, a, R_ref=0.05, name='', dk=None, kcut=40.0,
                 image_len=500.0, calibrate=True):
        self.n, self.skew, self.B_ref = int(n), bool(skew), float(B_ref)
        self.z0, self.z1, self.a, self.R_ref = float(z0), float(z1), float(a), float(R_ref)
        self.name = name
        # periodic images of the midpoint sum sit 4 pi / dk apart in z
        self.dk = dk if dk is not None else 4 * np.pi / image_len
        self.kcut = float(kcut)
        n_ = self.n
        self.pref = -2.0 * self.B_ref * (self.a / self.R_ref) ** (n_ - 1) * self.a ** 2 / (n_ * np.pi)
        self.scale = 1.0
        if calibrate:
            self.scale = calibration_scale(self)
            self.pref *= self.scale

    def Q(self, r, z, zchunk=64):
        '''(Q, dQ/dr, dQ/dz, nQ/r) for scalar r (< a) and array z'''
        a, n = self.a, self.n
        if r >= a:
            raise ValueError('r = %g must be inside the sheet radius a = %g' % (r, a))
        kmax = self.kcut / (a - r)
        k = (np.arange(int(np.ceil(kmax / self.dk))) + 0.5) * self.dk
        dk = self.dk
        # K_n'(ka) I_n(kr) and relatives, with the common e^(-k(a-r)) taken out of the scaled Bessels
        ka, kr = k * a, k * r
        Kp = -0.5 * (kve(n - 1, ka) + kve(n + 1, ka))
        damp = np.exp(-k * (a - r))
        In = ive(n, kr)
        Inp = 0.5 * (ive(n - 1, kr) + ive(n + 1, kr))
        nIn_over_kr = 0.5 * (ive(n - 1, kr) - ive(n + 1, kr))   # (n / (k r)) I_n(k r), finite at r = 0
        w_Q = self.pref * Kp * In * damp * dk
        w_r = self.pref * Kp * k * Inp * damp * dk
        w_n = self.pref * Kp * k * nIn_over_kr * damp * dk
        w_z = self.pref * Kp * In * k * damp * dk
        z = np.atleast_1d(np.asarray(z, float))
        out = [np.empty(len(z)) for _ in range(4)]
        for i in range(0, len(z), zchunk):
            zz = z[i:i + zchunk]
            u = np.outer(zz - self.z0, k)
            v = np.outer(zz - self.z1, k)
            S = np.sin(u) - np.sin(v)
            Cz = np.cos(u) - np.cos(v)
            out[0][i:i + zchunk] = S @ w_Q
            out[1][i:i + zchunk] = S @ w_r
            out[2][i:i + zchunk] = Cz @ w_z
            out[3][i:i + zchunk] = S @ w_n
        return tuple(out)


class SquareLoop(object):
    '''Thin-filament square current loop with moment m [A m^2] along moment_dir (exact Biot-Savart).

    The same corners, circulation and current (I = m / side^2) as helicalc.multipole.make_square_loop,
    whose bars have a 2 x 3.75 mm cross-section; at the >= 40 mm from the loop where the field is used,
    the filament differs by ~(size/d)^2/24 ~ 1e-4 relative.  Straight segment A -> B at P (a = A - P,
    b = B - P): B = mu0 I / (4 pi) (|a| + |b|) (a x b) / (|a| |b| (|a| |b| + a . b)).
    '''

    def __init__(self, center, moment_dir, moment, side=0.04, name='PERT'):
        c = np.asarray(center, float)
        nhat = np.asarray(moment_dir, float)
        nhat = nhat / np.linalg.norm(nhat)
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, nhat)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        u = np.cross(nhat, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(nhat, u)
        h = 0.5 * side
        self.corners = np.array([c - h * u - h * v, c + h * u - h * v, c + h * u + h * v, c - h * u + h * v])
        self.I = moment / side ** 2
        self.name = name
        self.n, self.skew = 0, False

    def moment(self):
        '''m = I/2 sum r x dl over the closed polygon (check)'''
        P = self.corners
        return 0.5 * self.I * sum(np.cross(P[i], P[(i + 1) % 4] - P[i]) for i in range(4))

    def B_xyz(self, x, y, z):
        P = np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)], axis=-1)
        B = np.zeros_like(P)
        for i in range(4):
            a = self.corners[i] - P
            b = self.corners[(i + 1) % 4] - P
            na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
            f = (na + nb) / (na * nb * (na * nb + np.sum(a * b, axis=-1)))
            B += f[..., None] * np.cross(a, b)
        B *= MU0 * self.I / (4 * np.pi)
        return B[..., 0], B[..., 1], B[..., 2]


def pert_loop(**kw):
    '''The handoff PERT (helicalc.multipole.PERT_SPEC): 40 mm loop, m = 15 A m^2 along y at (0.12, 0, 2.15)'''
    from helicalc.multipole import PERT_SPEC as s
    return SquareLoop(s['center'], s['moment_dir'], s['moment'], s['side'], name=s['name'], **kw)


# ---------------------------------------------------------------------------
# field evaluation on points
# ---------------------------------------------------------------------------
def field_from_Q(elem, r, phi, Qs):
    '''(Bx, By, Bz) from (Q, dQ/dr, dQ/dz, nQ/r) at polar (r, phi)'''
    Q, dQr, dQz, nQr = Qs
    n = elem.n
    s, c = np.sin(n * phi), np.cos(n * phi)
    if not elem.skew:        # psi = -Q sin(n phi)
        Br, Bphi, Bz = dQr * s, nQr * c, dQz * s
    else:                    # psi = -Q cos(n phi)
        Br, Bphi, Bz = dQr * c, -nQr * s, dQz * c
    cp, sp = np.cos(phi), np.sin(phi)
    return Br * cp - Bphi * sp, Br * sp + Bphi * cp, Bz


def add_field(df, elements, r_round=9, per_element=False, **qkw):
    '''Bx, By, Bz [T] at df.X, df.Y, df.Z summed over `elements` (returns a copy).

    Points are grouped by radius (rounded to r_round decimals), so Q is evaluated once per
    (r, unique z) and reused for every phi.
    '''
    out = df.copy()
    X, Y, Z = (np.asarray(df[c], float) for c in ('X', 'Y', 'Z'))
    r = np.round(np.hypot(X, Y), r_round)
    phi = np.arctan2(Y, X)
    tot = [np.zeros(len(df)) for _ in range(3)]
    for el in elements:
        part = [np.zeros(len(df)) for _ in range(3)]
        if hasattr(el, 'B_xyz'):          # conductor elements (SquareLoop): direct Biot-Savart
            part = list(el.B_xyz(X, Y, Z))
        for rv in (np.unique(r) if not hasattr(el, 'B_xyz') else []):
            sel = np.where(r == rv)[0]
            zu, inv = np.unique(Z[sel], return_inverse=True)
            Qs = el.Q(float(rv), zu, **qkw)
            Qs = tuple(q[inv] for q in Qs)
            b = field_from_Q(el, rv, phi[sel], Qs)
            for j in range(3):
                part[j][sel] = b[j]
        for j in range(3):
            tot[j] += part[j]
        if per_element:
            for j, c in enumerate(('Bx', 'By', 'Bz')):
                out['%s_%s' % (c, el.name)] = part[j]
    for j, c in enumerate(('Bx', 'By', 'Bz')):
        out[c] = tot[j]
    return out


def field_xyz(elements, x, y, z, **qkw):
    '''(Bx, By, Bz) at arrays of points, one point at a time per unique (r, z) -- for checks'''
    import pandas as pd
    df = pd.DataFrame({'X': np.atleast_1d(x), 'Y': np.atleast_1d(y), 'Z': np.atleast_1d(z)})
    o = add_field(df, elements, **qkw)
    return o.Bx.values, o.By.values, o.Bz.values


def nested_dipoles(model='B', enge_H=None, enge_V=None, aperture=0.150, a=0.090, **kw):
    '''MCBH (normal) + MCBV (skew, nested 10 mm outside), 2.05 T each, z 2.4-4.6 m.

    Strengths, z edges and the MCBV offset come from helicalc.multipole.ASSEMBLY_ELEMENTS, R_ref
    and D from MultipoleGeom(aperture).  a = winding radius of the inner coil; the default
    0.090 m is what the helicalc saddle geometry was built with (--winding-radius 0.090), so
    the sheets sit on the helicalc bar centres (MCBH 90 mm, MCBV 100 mm).
    '''
    from helicalc.multipole import ASSEMBLY_ELEMENTS, MultipoleGeom
    g = MultipoleGeom(aperture=aperture, a=a)
    spec = {s['name']: s for s in ASSEMBLY_ELEMENTS}
    els = []
    for name, eng in (('MCBH', enge_H), ('MCBV', enge_V)):
        s = spec[name]
        a_el = g.a + s.get('a_offset', 0.0)
        if model == 'B':
            els.append(SheetMultipole(s['n'], s['skew'], s['B_ref'], s['z0'], s['z1'], a_el,
                                      R_ref=g.R_ref, name=name, **kw))
        else:
            els.append(GGMultipole(s['n'], s['skew'], s['B_ref'], s['z0'], s['z1'], R_ref=g.R_ref,
                                   D=g.aperture, enge=eng if eng is not None else (0.0, 4.0),
                                   name=name, **kw))
    return els
