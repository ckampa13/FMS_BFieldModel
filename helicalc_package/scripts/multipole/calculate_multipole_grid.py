'''
Calculate the multipole assembly field on a grid, on one GPU.

Unlike helicalc's Mu2e bus-bar scripts (one process per conductor, one pickle
each) this sums all assigned bars in-process, because a multipole assembly has
~1100 of them.  Split across GPUs by BAR index with -D/-N; the partial pickles
are summed by sum_multipole_partials.py.

    python calculate_multipole_grid.py -r body -t y
    python calculate_multipole_grid.py -r map -D 0 -N 4
'''
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from helicalc import helicalc_data
from helicalc.multipole import (MultipoleGeom, load_assembly_csv, MAPPING_VOLUME,
                                ASSEMBLY_ELEMENTS)
from helicalc.multipole_field import add_field, DXYZ_MULTIPOLE
from helicalc.tools import generate_cartesian_grid_df, add_points_for_J

OUTDIR = os.path.join(helicalc_data, 'Bmaps', 'multipole', '')


def make_region(name, geom):
    '''Field-point DataFrame for a named region.'''
    R, z0, z1 = geom.R_map, MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1']
    if name == 'map':
        # transverse-plane grid inside the mapping cylinder, over the full length
        d = 0.010
        g = {'X0': -R, 'Y0': -R, 'Z0': z0, 'dX': d, 'dY': d, 'dZ': 0.020,
             'nX': int(2 * R / d) + 1, 'nY': int(2 * R / d) + 1,
             'nZ': int((z1 - z0) / 0.020) + 1}
        df = generate_cartesian_grid_df(g, dec_round=6)
        return df[np.hypot(df.X, df.Y) <= R + 1e-9].reset_index(drop=True)
    if name == 'body':
        # circles at R_ref through each element centre: the rotating-coil reference
        M, rows = 256, []
        ph = 2 * np.pi * np.arange(M) / M
        for spec in ASSEMBLY_ELEMENTS:
            zc = 0.5 * (spec['z0'] + spec['z1'])
            rows.append(pd.DataFrame({'X': geom.R_ref * np.cos(ph),
                                      'Y': geom.R_ref * np.sin(ph),
                                      'Z': np.full(M, zc),
                                      'HP': spec['name']}))
        return pd.concat(rows, ignore_index=True)
    if name == 'axis':
        # on-axis and an off-axis helix, for effective-length profiles
        zs = np.arange(z0, z1 + 1e-9, 0.005)
        rows = [pd.DataFrame({'X': 0.0, 'Y': 0.0, 'Z': zs, 'HP': 'axis'})]
        for lab, ang in (('x', 0.0), ('y', np.pi / 2)):
            rows.append(pd.DataFrame({'X': geom.R_ref * np.cos(ang),
                                      'Y': geom.R_ref * np.sin(ang),
                                      'Z': zs, 'HP': 'r_ref_' + lab}))
        return pd.concat(rows, ignore_index=True)
    if name == 'maxwell':
        # random points well inside the mapping cylinder, for div/curl checks
        rng = np.random.default_rng(12345)
        n = 2000
        rr = 0.9 * R * np.sqrt(rng.random(n))
        pp = 2 * np.pi * rng.random(n)
        return pd.DataFrame({'X': rr * np.cos(pp), 'Y': rr * np.sin(pp),
                             'Z': rng.uniform(z0 + 0.05, z1 - 0.05, n)})
    raise ValueError("region must be one of: map, body, axis, maxwell")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-r', '--Region', default='body',
                   help='map | body (default) | axis | maxwell')
    p.add_argument('-D', '--Device', type=int, default=0, help='GPU index')
    p.add_argument('-N', '--NDevices', type=int, default=1,
                   help='total GPUs the bar list is split across')
    p.add_argument('-g', '--Geom', default='Multipole_HLLHC_V1',
                   help='assembly basename in dev/params/')
    p.add_argument('-e', '--Element', default=None,
                   help='restrict to one element (e.g. MQ)')
    p.add_argument('-j', '--Jacobian', default='n',
                   help='add points for the Jacobian? y/n(default)')
    p.add_argument('-d', '--dxyz_Jacobian', type=float, default=0.001)
    p.add_argument('-t', '--Testing', default='n',
                   help='small subset of field points? y/n(default)')
    p.add_argument('--aperture', type=float, default=0.150)
    p.add_argument('--winding-radius', default='0.090')
    p.add_argument('--per-element', action='store_true')
    p.add_argument('--log', action='store_true', help='redirect stdout to a log file')
    args = p.parse_args(argv)

    a = None if str(args.winding_radius).lower() in ('auto', 'none', '') \
        else float(args.winding_radius)
    geom = MultipoleGeom(aperture=args.aperture, a=a)

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, 'logs'), exist_ok=True)
    old_stdout = sys.stdout
    if args.log:
        dt = datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M%S')
        sys.stdout = open(os.path.join(
            OUTDIR, 'logs', '%s_multipole_%s_GPU%d.log'
            % (dt, args.Region, args.Device)), 'w')

    try:
        bars = load_assembly_csv(args.Geom)
        if args.Element:
            bars = bars[bars.element == args.Element].reset_index(drop=True)
        # split bars across GPUs, heaviest first so the load balances
        if args.NDevices > 1:
            order = np.argsort(-bars['length'].values)
            mine = order[args.Device::args.NDevices]
            bars = bars.iloc[np.sort(mine)].reset_index(drop=True)

        df = make_region(args.Region, geom)
        if args.Testing.strip() == 'y':
            df = df.iloc[:500].copy().reset_index(drop=True)
        suff = ''
        if args.Jacobian.strip() == 'y':
            df = add_points_for_J(df, dxyz=args.dxyz_Jacobian)
            suff = '_Jacobian'

        print('region=%s  points=%d  bars=%d  GPU=%d/%d'
              % (args.Region, len(df), len(bars), args.Device, args.NDevices))
        out = add_field(df, bars, dev=args.Device, per_element=args.per_element)

        tag = '' if args.Element is None else '_' + args.Element
        name = '%s.%s_region%s%s.GPU%d_of_%d.pkl' % (
            args.Geom, args.Region, tag, suff, args.Device, args.NDevices)
        path = os.path.join(OUTDIR, name)
        out.to_pickle(path)
        print('wrote %s' % path)
    finally:
        if args.log:
            sys.stdout.close()
            sys.stdout = old_stdout
    return path


if __name__ == '__main__':
    print(main())
