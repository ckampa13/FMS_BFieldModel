'''
Harmonic rings: b_n / a_n of the assembly, the "rotating coil" measurement.

Samples the field on circles of radius R_ref and Fourier-analyses each one.
Two z modes:

  centres  one ring at each element's centre -- the reference table for the paper
  scan     rings every --dz along the mapping volume -- harmonics as a function
           of z, which is what shows the fringe and the neighbour cross-talk

Do NOT read harmonics off the Cartesian map grid: at 20 mm it has at most 8
azimuths on any one radius, so it resolves n <= 3, and this assembly goes to
n = 6.  A 256-point ring resolves n <= 127.

    python calculate_harmonic_rings.py -g Multipole_HLLHC_V1_saddle
    python calculate_harmonic_rings.py -g ... --z-mode scan --dz 0.02 -D 0 -N 4
    python calculate_harmonic_rings.py --analyze-only <summed.pkl>
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd

from helicalc import helicalc_data
from helicalc.multipole import (MultipoleGeom, load_assembly_csv,
                                ASSEMBLY_ELEMENTS, MAPPING_VOLUME,
                                make_ring_grid, harmonics_dataframe)
from helicalc.multipole_field import add_field

OUTDIR = os.path.join(helicalc_data, 'Bmaps', 'multipole', '')


def ring_zs(mode, dz, geom):
    '''(z positions, labels) for the requested mode.'''
    if mode == 'centres':
        zs = [0.5 * (s['z0'] + s['z1']) for s in ASSEMBLY_ELEMENTS]
        return zs, [s['name'] for s in ASSEMBLY_ELEMENTS]
    if mode == 'scan':
        z0, z1 = MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1']
        zs = np.arange(z0, z1 + 1e-9, dz)
        return list(zs), ['z%08.4f' % z for z in zs]
    raise ValueError("z-mode must be 'centres' or 'scan'")


def summarise(h, geom):
    '''Print the reference table: main harmonic and the significant others.'''
    print('  %-10s %-8s %-6s %-12s %-s'
          % ('ring', 'z [m]', 'n_main', 'main [T]', 'others > 0.5 units'))
    for ring, sub in h.groupby('ring', sort=False):
        nm = int(sub['n_main'].iloc[0])
        main = sub[sub.n == nm]
        val = float(np.hypot(main['B_n'].iloc[0], main['A_n'].iloc[0]))
        big = []
        for _, r in sub.iterrows():
            if int(r['n']) == nm:
                continue
            for lab, v in (('b', r['b_n']), ('a', r['a_n'])):
                if abs(v) > 0.5:
                    big.append('%s%d=%+.1f' % (lab, int(r['n']), v))
        print('  %-10s %-8.3f %-6d %-12.5f %s'
              % (ring, sub['z'].iloc[0], nm, val,
                 ' '.join(big[:8]) or '(none)'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-g', '--Geom', default='Multipole_HLLHC_V1_saddle')
    p.add_argument('-D', '--Device', type=int, default=0)
    p.add_argument('-N', '--NDevices', type=int, default=1,
                   help='split the CONDUCTORS across this many GPUs')
    p.add_argument('-M', '--NPhi', type=int, default=256,
                   help='points per ring (resolves n <= M/2-1; default 256)')
    p.add_argument('-r', '--Radius', type=float, default=None,
                   help='ring radius [m]; default geom.R_ref')
    p.add_argument('--z-mode', default='centres', choices=['centres', 'scan'])
    p.add_argument('--dz', type=float, default=0.020, help='z step for scan [m]')
    p.add_argument('--n-max', type=int, default=20)
    p.add_argument('--aperture', type=float, default=0.150)
    p.add_argument('--winding-radius', default='0.090')
    p.add_argument('--per-element', action='store_true',
                   help='also keep each winding separately (isolated harmonics)')
    p.add_argument('--analyze-only', default=None,
                   help='skip the field calculation; analyse this summed pickle')
    args = p.parse_args(argv)

    a = None if str(args.winding_radius).lower() in ('auto', 'none', '') \
        else float(args.winding_radius)
    geom = MultipoleGeom(aperture=args.aperture, a=a)
    R = args.Radius if args.Radius is not None else geom.R_ref
    os.makedirs(OUTDIR, exist_ok=True)

    if args.analyze_only:
        out = pd.read_pickle(args.analyze_only)
        base = os.path.basename(args.analyze_only).replace('.pkl', '')
    else:
        bars = load_assembly_csv(args.Geom)
        if args.NDevices > 1:
            order = np.argsort(-bars['length'].values)
            bars = bars.iloc[np.sort(order[args.Device::args.NDevices])
                             ].reset_index(drop=True)
        zs, labels = ring_zs(args.z_mode, args.dz, geom)
        df = make_ring_grid(zs, R, M=args.NPhi, labels=labels)
        print('rings: %d at r=%.1f mm, %d points each, %d total; bars=%d GPU=%d/%d'
              % (len(zs), 1e3 * R, args.NPhi, len(df), len(bars),
                 args.Device, args.NDevices))
        out = add_field(df, bars, dev=args.Device,
                        per_element=args.per_element)
        base = '%s.rings_%s_r%.0fmm' % (args.Geom, args.z_mode, 1e3 * R)
        if args.NDevices > 1:
            path = os.path.join(OUTDIR, '%s.GPU%d_of_%d.pkl'
                                % (base, args.Device, args.NDevices))
            out.to_pickle(path)
            print('wrote %s' % path)
            print('sum the partials, then re-run with --analyze-only')
            return path
        out.to_pickle(os.path.join(OUTDIR, base + '.pkl'))

    h = harmonics_dataframe(out, n_max=args.n_max)
    hp = os.path.join(OUTDIR, base + '.harmonics.csv')
    h.to_csv(hp, index=False)
    print()
    summarise(h, geom)
    print()
    print('wrote %s  (%d rings x %d orders)' % (hp, h['ring'].nunique(), args.n_max))
    return hp


if __name__ == '__main__':
    main()
