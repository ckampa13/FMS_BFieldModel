'''
Sum helicalc's straight-bar Biot-Savart integrator over a whole multipole
assembly, in one process.

helicalc ships one-conductor-per-process drivers that each write a pickle, which
is fine for Mu2e's ~44 bus bars but not for the ~1100 bars of a multipole
assembly.  This module loops bars in-process and accumulates the field.

Split from multipole.py because this half needs torch/CUDA while the geometry
half deliberately does not.
'''
from time import time
import numpy as np
import pandas as pd
import torch as tc
from tqdm import tqdm

from .busbar import StraightIntegrator3D, ArcIntegrator3D
from .constants import MAXMEM
from . import tools as _tools

# cross-section sampling for a multipole bar (W=8mm, T=15mm by default):
# 5 x 6 nodes across the conductor, 5 mm along it.
DXYZ_MULTIPOLE = np.array([2e-3, 3e-3, 5e-3])

# Arcs need a finer step than straights at the same linear resolution: a corner
# fillet is short, tightly curved and sits close to the mapping volume, so its
# integrand varies far faster along the element than a 1.5 m body bar's does.
# Measured on the MQ winding, the curl-free residual at the element end falls
# 0.84 -> 0.19 -> 0.046 -> 0.011 G as this is halved, i.e. clean h^2 convergence
# TO ZERO (contrast the chord closure, which converges to a nonzero 4.4 G).
# A quarter of the straight-bar step puts it comfortably under the 0.3 G noise.
DXYZ_MULTIPOLE_ARC = np.array([0.5e-3, 0.75e-3, 1.25e-3])

# bytes per element and number of (N_batch, nx, ny, nz) arrays that
# StraightIntegrator3D.integrate_vec holds live at once
_BYTES = 8
# RX, RY, RZ, R2_32, the integrand, and the intermediates trapz_3d builds while
# reducing three axes.  Measured empirically: 6 is optimistic and OOMs on an
# 11 GB card at fine dxyz, so budget for 9.
_N_LIVE = 9


# ----------------------------------------------------------------------------
# nvidia-smi memoisation
# ----------------------------------------------------------------------------
# StraightIntegrator3D.__init__ calls get_gpu_memory_map() unconditionally, which
# shells out to nvidia-smi.  Constructing ~1100 integrators would spawn ~1100
# subprocesses (seconds of pure overhead, and it can trip process limits), so the
# result is cached for the life of the process.
_GPU_MEM_CACHE = {'value': None}
_ORIG_GET_GPU_MEMORY_MAP = _tools.get_gpu_memory_map


def _cached_get_gpu_memory_map():
    if _GPU_MEM_CACHE['value'] is None:
        _GPU_MEM_CACHE['value'] = _ORIG_GET_GPU_MEMORY_MAP()
    return _GPU_MEM_CACHE['value']


def enable_gpu_memory_cache():
    '''Memoise helicalc.tools.get_gpu_memory_map (and the copy busbar imported).'''
    _tools.get_gpu_memory_map = _cached_get_gpu_memory_map
    import helicalc.busbar as _bb
    _bb.get_gpu_memory_map = _cached_get_gpu_memory_map


def clear_gpu_memory_cache():
    _GPU_MEM_CACHE['value'] = None


# ----------------------------------------------------------------------------
# batch sizing
# ----------------------------------------------------------------------------
def is_arc(row):
    return 'kind' in row.index and row['kind'] == 'arc'


def arc_mask(df):
    '''Boolean Series marking arc rows; all-False when there is no `kind` column.

    df.get('kind', 'straight') returns a SCALAR for a frame without the column,
    which then indexes the frame with a bare bool and raises -- so build the mask
    explicitly.
    '''
    if 'kind' in df.columns:
        return df['kind'].values == 'arc'
    return np.zeros(len(df), dtype=bool)


def arc_dxyz(row, dxyz):
    '''helicalc's arc integrator reads dxyz[2] as d(phi) ASSUMING R = 1 m, so it
    has to be divided by the actual bend radius to get the intended arc length
    step.  Getting this wrong silently changes the integration density.'''
    d = np.asarray(dxyz, float).copy()
    d[2] = d[2] / float(row['R_curve'])
    return d


def integration_node_count(row, dxyz):
    '''(nx, ny, nz) integration nodes helicalc will build for one element.'''
    nx = abs(int(row['W'] / dxyz[0] + 1))
    ny = abs(int(row['T'] / dxyz[1] + 1))
    if is_arc(row):
        dphi = np.radians(float(row['dphi']))
        nz = abs(int(dphi / arc_dxyz(row, dxyz)[2] + 1))
    else:
        nz = abs(int(row['length'] / dxyz[2] + 1))
    return nx, ny, nz


def resolve_dxyz(row, dxyz, min_nodes=3):
    '''Per-bar integration steps, refined so every dimension gets min_nodes.

    A multipole assembly mixes bar sizes by more than an order of magnitude (a
    2.2 m body bar and a 40 mm perturbation-loop side), and helicalc silently
    integrates to ZERO when a dimension gets a single node.  Rather than force
    one global dxyz, each bar's steps are shrunk (never coarsened) so that
    int(size/step + 1) >= min_nodes holds in all three dimensions.
    '''
    size = np.array([row['W'], row['T'], row['length']], float)
    need = size / max(min_nodes - 1, 1)
    return np.minimum(np.asarray(dxyz, float), need)


def build_integrator(row, dxyz, dev, lib, int_func, min_nodes=3):
    '''StraightIntegrator3D or ArcIntegrator3D, with the right dxyz for each.'''
    d = resolve_dxyz(row, dxyz, min_nodes)
    if is_arc(row):
        return ArcIntegrator3D(row, dxyz=arc_dxyz(row, d), dev=dev, lib=lib,
                               int_func=int_func)
    return StraightIntegrator3D(row, dxyz=d, dev=dev, lib=lib, int_func=int_func)


def check_integration_nodes(df, dxyz, min_nodes=3, adapt=True):
    '''Raise if any bar would get too few integration nodes.

    helicalc uses linspace(-W/2, W/2, int(W/dxyz[0] + 1)); when that count is 1
    the trapezoid integral over the dimension is silently ZERO, so a too-coarse
    dxyz produces a wrong answer with no error at all.

    With adapt=True (the default, matching add_field) the per-bar refinement of
    resolve_dxyz is applied first, so this only fails on a genuinely impossible
    request (e.g. a zero-size bar).
    '''
    bad = []
    for _, row in df.iterrows():
        d = resolve_dxyz(row, dxyz, min_nodes) if adapt else dxyz
        nx, ny, nz = integration_node_count(row, d)
        if min(nx, ny, nz) < min_nodes:
            bad.append((int(row['cond N']), nx, ny, nz))
    if bad:
        raise ValueError(
            'dxyz=%s gives too few integration nodes for %d bar(s), e.g. '
            'cond N=%d -> (nx,ny,nz)=(%d,%d,%d); need >= %d in each dimension'
            % (list(dxyz), len(bad), bad[0][0], bad[0][1], bad[0][2], bad[0][3],
               min_nodes))
    return True


def estimate_batch_size(row, dxyz, mem_frac=0.45, max_batch=200000, min_batch=4,
                       min_nodes=3):
    '''Field points per chunk that fit in mem_frac of one GPU.

    integrate_vec materialises ~6 float64 arrays of shape (N_batch, nx, ny, nz).
    '''
    nx, ny, nz = integration_node_count(row, resolve_dxyz(row, dxyz, min_nodes))
    nodes = max(nx * ny * nz, 1)
    budget = mem_frac * MAXMEM * 1024 ** 2          # MAXMEM is in MB
    n = int(budget / (_BYTES * _N_LIVE * nodes))
    return int(np.clip(n, min_batch, max_batch))


# ----------------------------------------------------------------------------
# the aggregator
# ----------------------------------------------------------------------------
def promote_nodes_to_float64(integ):
    '''Re-cast an integrator's integration nodes to float64.

    helicalc builds them with tc.linspace, which defaults to float32, so node
    positions carry ~1e-7 relative error (about 70 nm on a 1.5 m bar).

    This was TESTED as a candidate cause of the Maxwell curl floor documented in
    add_field and found NOT to be responsible: promoting to float64 reproduces
    the float32 residuals to every printed digit.  Kept as an option (it is
    nearly free) but off by default, so nobody re-runs the same experiment.
    '''
    integ.xps = integ.xps.double()
    integ.yps = integ.yps.double()
    integ.zps = integ.zps.double()
    integ.XP, integ.YP, integ.ZP = tc.meshgrid(integ.xps, integ.yps, integ.zps,
                                               indexing='ij')
    return integ


def add_field(df_points, df_bars, dxyz=None, dev=0, N_batch=None, mem_frac=0.45,
              per_element=False, units='T', tqdm=tqdm, lib=tc, int_func=None,
              prefix='B', verbose=True, check_nodes=True, min_nodes=3,
              float64_nodes=False, dxyz_arc=None):
    '''Sum StraightIntegrator3D over every bar in df_bars.

    df_points : DataFrame with X, Y, Z columns [m] (HP is carried through).
    df_bars   : DataFrame of straight bars (helicalc.multipole.build_assembly).
    dxyz      : integration steps; defaults to DXYZ_MULTIPOLE.
    N_batch   : field points per chunk; None -> estimate_batch_size per bar.
    per_element : also keep one Bx/By/Bz triple per `element` label.
    units     : 'T' (default) or 'G'.

    Returns a copy of df_points with prefix+'x/y/z' columns added.

    KNOWN LIMITATION -- spurious curl at conductor corners.
    helicalc models a bar as a rectangular prism whose current enters and leaves
    through flat faces perpendicular to that bar's own axis.  Where two bars of a
    loop meet at an angle the two faces are not coplanar, so the volumetric
    current is not conserved in the wedge at the joint.  The resulting field is
    still divergence-free -- div B converges as h^2 and matches the thin-wire
    model exactly -- but curl B does NOT vanish in the bore: it converges to a
    constant, about 0.03 T/m per loop, at points near an element's end turns,
    where it should be zero.

    Established by experiment, not assumed:
      - the thin-wire field of the same (Kirchhoff-closed to 0 A) geometry has
        div and curl both converging as h^2, ratio 4.0, with no floor;
      - the floor is unchanged by refining dxyz over a factor of 8;
      - it is unchanged by float32 vs float64 integration nodes;
      - one closed loop shows it near its corners but not at the magnet centre,
        where the two ends' contributions cancel.

    Confirmed by construction: an exactly-closed bundle of translated closed
    filaments with the SAME 8 x 15 mm cross-section is curl-free to 3e-4 G at the
    same probe, while helicalc's prism model of that conductor gives 0.36 G.  So
    the finite cross-section is not the problem -- the unmated end faces are.

    USE closure='saddle'.  It is the cos(n.theta) topology (loop magnetic moment
    radial, m.rhat = 1.000) with every joint arc-matched, and its residual
    converges to ZERO rather than to chord closure's 4.4 G floor:

        dxyz scale     1.0      0.5      0.25
        r = 30 mm     0.523    0.107    0.028  G
        r = 50 mm     1.410    0.342    0.088  G   (mapping boundary, worst case)

    i.e. clean h^2 at both radii.  A quarter of the default step puts the worst
    point at 0.088 G, well under the 0.3 G Hall-probe noise.

    FIX (verified): put an ARC element at each corner, with the adjoining straight
    bars' cross-section oriented so their faces mate.  An arc's end faces are
    perpendicular to its local tangent, so they mate exactly with a straight bar
    carrying the same Euler angles; helicalc's arc integrand already carries the
    (R - y') Jacobian, so the current stays uniform and divergence-free through
    the bend.  On a test racetrack this cut the residual from 0.504 G to 0.008 G
    at 50 mm and from 0.877 G to 0.033 G at 30 mm -- 27x to 67x, and well under
    the 0.3 G noise -- at the same winding radius and cross-section.

    The mating is the fiddly part: helicalc's arc bends in its local y-z plane, so
    T (not W) must lie in the bend plane, and psi2 on the straight bars must be set
    to match rather than left to psi_mode='radial'/'normal'.  Getting that wrong
    silently reintroduces the artefact at full size -- it did here on the first try.

    Second mitigation (verified, independent): moving the winding radius out.
    Measured at r = 50 mm, a = 90 -> 200 mm takes the MQ end from 9.65 to 1.03 G
    and every other element below 0.32 G; a = 300 mm reaches 0.34 G at the quad.
    It costs current (~a^n) and lengthens the fringe, which is the physics under
    test, so the arc corners are the better fix.

    Rejected by measurement: subdividing bars into n x n sub-bars (no effect --
    sub-bars in the same local frame tile the identical prism), shrinking the
    cross-section (non-monotonic), and rounding corners with extra STRAIGHT
    segments (slightly worse -- more joints, more wedges).
    '''
    if dxyz is None:
        dxyz = DXYZ_MULTIPOLE
    dxyz = np.asarray(dxyz, float)
    if dxyz_arc is None:
        # scale the arc default the same way the caller scaled the straight one
        dxyz_arc = DXYZ_MULTIPOLE_ARC * (dxyz / DXYZ_MULTIPOLE)
    dxyz_arc = np.asarray(dxyz_arc, float)
    if int_func is None:
        int_func = tc.trapz if lib is tc else np.trapz
    if check_nodes:
        m = arc_mask(df_bars)
        straights = df_bars[~m]
        arcs = df_bars[m]
        if len(straights):
            check_integration_nodes(straights, dxyz, min_nodes=min_nodes)
        if len(arcs):
            check_integration_nodes(arcs, dxyz_arc, min_nodes=min_nodes)
    enable_gpu_memory_cache()

    scale = 1e4 if units == 'G' else 1.0
    if units not in ('T', 'G'):
        raise ValueError("units must be 'T' or 'G'")

    df = df_points.copy()
    npts = len(df)
    xyz = df[['X', 'Y', 'Z']].values.astype(float)
    total = np.zeros((3, npts))
    per_el = {}
    if per_element and 'element' in df_bars.columns:
        for el in df_bars['element'].unique():
            per_el[el] = np.zeros((3, npts))

    t0 = time()
    it = df_bars.itertuples(index=False)
    bar_iter = tqdm(it, total=len(df_bars), desc='bar') if tqdm is not None else it
    for rec in bar_iter:
        row = pd.Series(rec._asdict())
        if float(row['I']) == 0.0:
            continue
        d_row = dxyz_arc if is_arc(row) else dxyz
        nb = N_batch or estimate_batch_size(row, d_row, mem_frac=mem_frac,
                                            min_nodes=min_nodes)
        integ = build_integrator(row, d_row, dev, lib, int_func, min_nodes)
        if float64_nodes and lib is tc and not is_arc(row):
            promote_nodes_to_float64(integ)
        acc = np.zeros((3, npts))
        for s in range(0, npts, nb):
            e = min(s + nb, npts)
            B = integ.integrate_vec(xyz[s:e, 0], xyz[s:e, 1], xyz[s:e, 2])
            acc[:, s:e] = np.asarray(B)
        total += acc
        if per_el:
            per_el[row['element']] += acc
        del integ
        if lib is tc:
            tc.cuda.empty_cache()      # fine dxyz fragments the caching allocator

    for i, c in enumerate('xyz'):
        df[prefix + c] = scale * total[i]
    for el, arr in per_el.items():
        for i, c in enumerate('xyz'):
            df['%s%s_%s' % (prefix, c, el)] = scale * arr[i]
    if verbose:
        print('add_field: %d bars x %d points in %.1f s'
              % (len(df_bars), npts, time() - t0))
    return df
