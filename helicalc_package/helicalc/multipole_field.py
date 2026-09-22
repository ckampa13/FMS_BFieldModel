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

from .busbar import StraightIntegrator3D
from .constants import MAXMEM
from . import tools as _tools

# cross-section sampling for a multipole bar (W=8mm, T=15mm by default):
# 5 x 6 nodes across the conductor, 5 mm along it.
DXYZ_MULTIPOLE = np.array([2e-3, 3e-3, 5e-3])

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
def integration_node_count(row, dxyz):
    '''(nx, ny, nz) integration nodes helicalc will build for one bar.'''
    nx = abs(int(row['W'] / dxyz[0] + 1))
    ny = abs(int(row['T'] / dxyz[1] + 1))
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
              float64_nodes=False):
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

    It is a property of the prism conductor model rather than of this module, and
    it applies equally to helicalc's Mu2e bus bars.  Removing it needs a joint-
    matched conductor model (mitred end faces, or a filament bundle routed with a
    common cross-section offset through every bar of a loop).  Until then treat
    curl B near element ends as a model artefact, and use thin_wire_field when an
    exactly curl-free reference is needed.
    '''
    if dxyz is None:
        dxyz = DXYZ_MULTIPOLE
    dxyz = np.asarray(dxyz, float)
    if int_func is None:
        int_func = tc.trapz if lib is tc else np.trapz
    if check_nodes:
        check_integration_nodes(df_bars, dxyz, min_nodes=min_nodes)
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
        d_bar = resolve_dxyz(row, dxyz, min_nodes)
        nb = N_batch or estimate_batch_size(row, dxyz, mem_frac=mem_frac,
                                            min_nodes=min_nodes)
        integ = StraightIntegrator3D(row, dxyz=d_bar, dev=dev, lib=lib,
                                     int_func=int_func)
        if float64_nodes and lib is tc:
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
