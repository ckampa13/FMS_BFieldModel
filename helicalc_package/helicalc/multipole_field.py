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
_N_LIVE = 6


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


def estimate_batch_size(row, dxyz, mem_frac=0.6, max_batch=200000, min_batch=64,
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
def add_field(df_points, df_bars, dxyz=None, dev=0, N_batch=None, mem_frac=0.6,
              per_element=False, units='T', tqdm=tqdm, lib=tc, int_func=None,
              prefix='B', verbose=True, check_nodes=True, min_nodes=3):
    '''Sum StraightIntegrator3D over every bar in df_bars.

    df_points : DataFrame with X, Y, Z columns [m] (HP is carried through).
    df_bars   : DataFrame of straight bars (helicalc.multipole.build_assembly).
    dxyz      : integration steps; defaults to DXYZ_MULTIPOLE.
    N_batch   : field points per chunk; None -> estimate_batch_size per bar.
    per_element : also keep one Bx/By/Bz triple per `element` label.
    units     : 'T' (default) or 'G'.

    Returns a copy of df_points with prefix+'x/y/z' columns added.
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
        acc = np.zeros((3, npts))
        for s in range(0, npts, nb):
            e = min(s + nb, npts)
            B = integ.integrate_vec(xyz[s:e, 0], xyz[s:e, 1], xyz[s:e, 2])
            acc[:, s:e] = np.asarray(B)
        total += acc
        if per_el:
            per_el[row['element']] += acc
        del integ

    for i, c in enumerate('xyz'):
        df[prefix + c] = scale * total[i]
    for el, arr in per_el.items():
        for i, c in enumerate('xyz'):
            df['%s%s_%s' % (prefix, c, el)] = scale * arr[i]
    if verbose:
        print('add_field: %d bars x %d points in %.1f s'
              % (len(df_bars), npts, time() - t0))
    return df
