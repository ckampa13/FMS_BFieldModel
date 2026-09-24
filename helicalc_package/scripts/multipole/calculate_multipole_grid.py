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


def make_region(name, geom, dxy=0.020, dz=0.020, radii=None, nphi=32,
                dedupe_axis=True, offset=False, z_range=None, z_subdiv=1):
    '''Field-point DataFrame for a named region.

    dxy, dz         Cartesian grid steps for the 'map' region [m]
    radii, nphi, dz probe radii, azimuths per revolution and z step for the
                    'measurement' region
    offset          shift the 'map' grid by half a step in z, so that a test
                    sample does not sit on the same z planes as the measurement
                    set.  Without it, the propeller's phi = 0/90/180/270 arms at
                    r = 0/20/40/60 mm land exactly on a 20 mm cartesian grid at
                    the same z, and 45% of the test points are locations the fit
                    was trained on.
    dedupe_axis     an on-axis probe does not move as the propeller turns, so
                    its nphi azimuths are all the same point.  True keeps one
                    row per z for it (the truth field there is identical);
                    downstream, replicate that row nphi times if you want nphi
                    independent noisy readings.  False emits all nphi.
    z_range         (z0, z1) [m]; default MAPPING_VOLUME.  The 'map' grid keeps
                    its half-step offset relative to z0.
    z_subdiv        split each 'measurement' z step into this many planes
                    (default 1).  Must be odd: an even split puts a plane at
                    z0 + dz/2 (mod dz), i.e. on the offset 'map' test planes.
    '''
    R = geom.R_map
    z0, z1 = MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1']
    if z_range is not None:
        z0, z1 = float(z_range[0]), float(z_range[1])
        if not z1 > z0:
            raise ValueError('z_range must be increasing: %r' % (z_range,))
    z_subdiv = int(z_subdiv)
    if z_subdiv < 1:
        raise ValueError('z_subdiv must be >= 1')
    if name == 'measurement':
        # Propeller sampling (handoff sec 6): probes at fixed radii on a rotating
        # arm, stepped along z.  nphi = 32 matters -- with 16 the n = 10 content
        # aliases onto n = 6, and this assembly carries both.
        if radii is None:
            radii = [0.0, 0.020, 0.040, 0.060]
        radii = [r for r in radii if r <= R + 1e-9]
        if z_range is None and z_subdiv == 1:
            # legacy grid, kept bit-for-bit so existing datasets stay comparable
            zs = np.arange(z0, z1 + 1e-9, dz)
        else:
            # integer plane index, no float-arange drift
            step = dz / z_subdiv
            n = int(np.floor((z1 - z0) / step + 1e-6)) + 1
            zs = np.round(z0 + np.arange(n) * step, 9)
            # no plane may sit on an offset 'map' test plane z0 + dz/2 + j*dz
            frac = np.mod(zs - z0 - 0.5 * dz, dz)
            clash = np.minimum(frac, dz - frac) < 1e-6
            if clash.any():
                raise ValueError('z_subdiv=%d puts %d measurement planes on the '
                                 'offset map planes (use an odd split)'
                                 % (z_subdiv, clash.sum()))
        ph = 2 * np.pi * np.arange(nphi) / nphi
        frames = []
        for r in radii:
            use = ph[:1] if (dedupe_axis and r <= 1e-12) else ph
            for z in zs:
                frames.append(pd.DataFrame({
                    'X': r * np.cos(use), 'Y': r * np.sin(use),
                    'Z': np.full(len(use), z),
                    'HP': 'r%03.0fmm' % (1e3 * r),
                    'phi': use,
                    'n_rep': (nphi if (dedupe_axis and r <= 1e-12) else 1)}))
        return pd.concat(frames, ignore_index=True)
    if name == 'map':
        # Transverse grid inside the mapping cylinder, over the full length.
        # Build it CENTRED on the axis: starting at -R with nX = int(2R/d)+1
        # only lands symmetrically when d divides 2R, and silently does not
        # otherwise -- a 25 mm step gives [-60, -35, -10, 15, 40] mm, which
        # misses the axis, is asymmetric, and never reaches +R.
        k = int(np.floor(R / dxy + 1e-9))
        zoff = 0.5 * dz if offset else 0.0
        nz = int(round((z1 - z0 - zoff) / dz)) + 1
        g = {'X0': -k * dxy, 'Y0': -k * dxy, 'Z0': z0 + zoff,
             'dX': dxy, 'dY': dxy, 'dZ': dz,
             'nX': 2 * k + 1, 'nY': 2 * k + 1, 'nZ': nz}
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
    raise ValueError("region must be one of: map, measurement, body, axis, "
                     "maxwell")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-r', '--Region', default='body',
                   help='map | measurement | body (default) | axis | maxwell')
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
    p.add_argument('--dxy', type=float, default=0.020,
                   help="transverse grid step for region 'map' [m] (default 0.020)")
    p.add_argument('--dz', type=float, default=0.020,
                   help="axial step for regions 'map' and 'measurement' [m]")
    p.add_argument('--radii', default='0,20,40,60',
                   help="probe radii in MM for region 'measurement' (default 0,20,40,60)")
    p.add_argument('--nphi', type=int, default=32,
                   help="azimuths per revolution for 'measurement' (default 32)")
    p.add_argument('--offset-map', action='store_true',
                   help="shift region 'map' by half a z step so the test sample "
                        'does not reuse measurement locations')
    p.add_argument('--z-range', default=None,
                   help='z0,z1 in M for the region (default: MAPPING_VOLUME, '
                        '0,8.6)')
    p.add_argument('--z-subdiv', type=int, default=1,
                   help="split each 'measurement' z step into K planes; K must "
                        'be odd (default 1)')
    p.add_argument('--name-tag', default='',
                   help='appended to the output name, e.g. Z1p9to5p1_k3_phi64, '
                        'so a non-default grid never overwrites a default one')
    p.add_argument('--keep-axis-copies', action='store_true',
                   help='emit all nphi rows for an on-axis probe instead of one')
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

        radii = [float(v) / 1e3 for v in args.radii.split(',') if v.strip()]
        z_range = None
        if args.z_range:
            z_range = [float(v) for v in args.z_range.split(',')]
            if len(z_range) != 2:
                raise SystemExit('--z-range takes z0,z1')
        df = make_region(args.Region, geom, dxy=args.dxy, dz=args.dz,
                         radii=radii, nphi=args.nphi,
                         dedupe_axis=not args.keep_axis_copies,
                         offset=args.offset_map, z_range=z_range,
                         z_subdiv=args.z_subdiv)
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
        if args.name_tag:
            tag += '_' + args.name_tag
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
