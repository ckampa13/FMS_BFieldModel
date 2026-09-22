'''
Build the HL-LHC-like multipole assembly and write it to dev/params/.

All physical dimensions are exposed on the command line; `--aperture` is the
primary knob and the winding radius, clearance, reference and mapping radii
follow from it unless overridden.

    python build_multipole_geometry.py                     # nominal assembly
    python build_multipole_geometry.py --aperture 0.100 --name Multipole_D100
    python build_multipole_geometry.py --quad-harmonics explicit
'''
import argparse
import numpy as np

from helicalc.multipole import (
    MultipoleGeom, build_assembly, save_assembly_csv, summarize_assembly,
    check_closure, min_conductor_radius, ASSEMBLY_ELEMENTS, MQ_HARMONIC_TARGETS,
    bars_per_pole, chord_segments,
)
from helicalc.multipole_field import DXYZ_MULTIPOLE, check_integration_nodes


def opt_float(v):
    """float, or None for 'auto' / 'none' / '' so a derived default is used."""
    if v is None or str(v).strip().lower() in ('auto', 'none', ''):
        return None
    return float(v)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--aperture', type=opt_float, default=0.150,
                   help='aperture DIAMETER [m] (default 0.150)')
    p.add_argument('--winding-radius', type=opt_float, default=0.090,
                   help="winding mean radius a [m]; 'auto' -> R_aper + T/2 "
                        '(default 0.090, the nominal HL-LHC-like value)')
    p.add_argument('--bar-W', type=opt_float, default=0.008,
                   help='bar cross-section, azimuthal [m] (default 0.008)')
    p.add_argument('--bar-T', type=opt_float, default=0.015,
                   help='bar cross-section, radial [m] (default 0.015)')
    p.add_argument('--R-ref', type=opt_float, default=None,
                   help='harmonic reference radius [m]; default R_aper/1.5')
    p.add_argument('--R-map', type=opt_float, default=None,
                   help='mapping radius [m]; default 0.8*R_aper')
    p.add_argument('--r-clear', type=opt_float, default=None,
                   help='min conductor radius [m]; default R_aper')
    p.add_argument('--N-target', type=int, default=32,
                   help='bars per winding before rounding to a multiple of 4n')
    p.add_argument('--closure', default='chord', choices=['chord', 'radial'],
                   help='winding closure (default chord)')
    p.add_argument('--return-radius', type=opt_float, default=None,
                   help="return radius b [m], required for --closure radial")
    p.add_argument('--quad-harmonics', default='sector',
                   choices=['sector', 'explicit', 'none'],
                   help='how the main quad gets its allowed harmonics')
    p.add_argument('--edge-mode', default='fractional',
                   choices=['fractional', 'binary'],
                   help='sector block edge treatment (default fractional)')
    p.add_argument('--corrector-scale', type=float, default=1.0,
                   help='scale factor on every corrector strength')
    p.add_argument('--layout', default='full',
                   help="'full', 'quad', or a comma-separated list of elements")
    p.add_argument('--no-pert', action='store_true',
                   help='omit the localized perturbation')
    p.add_argument('--name', default='Multipole_HLLHC_V1',
                   help='output basename in dev/params/')
    p.add_argument('--dry-run', action='store_true',
                   help='build and check but do not write the CSV')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    geom = MultipoleGeom(aperture=args.aperture, W=args.bar_W, T=args.bar_T,
                         a=args.winding_radius, R_ref=args.R_ref,
                         R_map=args.R_map, r_clear=args.r_clear,
                         N_target=args.N_target)
    layout = args.layout
    if layout not in ('full', 'quad'):
        layout = [s.strip() for s in layout.split(',') if s.strip()]
    qh = None if args.quad_harmonics == 'none' else args.quad_harmonics

    print('=' * 72)
    print('Geometry')
    print('=' * 72)
    print('  aperture        %8.1f mm (radius %.1f mm)' % (1e3 * geom.aperture,
                                                           1e3 * geom.R_aper))
    print('  winding radius  %8.1f mm  (bar inner face %.1f mm)'
          % (1e3 * geom.a, 1e3 * (geom.a - geom.T / 2)))
    print('  bar W x T       %8.1f x %.1f mm' % (1e3 * geom.W, 1e3 * geom.T))
    print('  R_ref / R_map   %8.1f / %.1f mm' % (1e3 * geom.R_ref, 1e3 * geom.R_map))
    print('  r_clear         %8.1f mm' % (1e3 * geom.r_clear))
    print('  N_target        %8d' % geom.N_target)
    print()
    print('  per order: bars N (multiple of 4n) and end-turn segments k')
    for n in range(1, 7):
        N = bars_per_pole(n, geom.N_target)
        k, _ = chord_segments(n, geom)
        print('    n=%d  N=%3d  k=%d  end-turn r_min=%.1f mm'
              % (n, N, k, 1e3 * geom.a * np.cos(np.pi / (2 * n * k))))
    print()

    df = build_assembly(layout=layout, corrector_scale=args.corrector_scale,
                        quad_harmonics=qh, geom=geom, closure=args.closure,
                        b=args.return_radius, include_pert=not args.no_pert,
                        edge_mode=args.edge_mode, verbose=True)

    print()
    print('=' * 72)
    print('Assembly')
    print('=' * 72)
    summary = summarize_assembly(df)
    print(summary.to_string(index=False, float_format=lambda v: '%.3f' % v))
    print()
    print('  total bars            %d' % len(df))
    print('  total conductor       %.1f m' % df['length'].sum())

    ok, worst, _ = check_closure(df)
    rmin = min_conductor_radius(df)
    check_integration_nodes(df, DXYZ_MULTIPOLE)
    print('  loops closed          %s (worst net current %.2e A)' % (ok, worst))
    print('  min conductor radius  %.1f mm  (R_map = %.1f mm, r_clear = %.1f mm)'
          % (1e3 * rmin, 1e3 * geom.R_map, 1e3 * geom.r_clear))
    print('  integration nodes     ok (>=3 per dimension after per-bar refinement)')

    problems = []
    if not ok:
        problems.append('winding is not closed')
    if rmin < geom.R_map:
        problems.append('a conductor enters the mapping volume')
    if rmin < geom.r_clear - 1e-9:
        problems.append('a conductor is inside r_clear')
    if df['cond N'].duplicated().any():
        problems.append('duplicate cond N')
    if df[['length', 'W', 'T', 'I', 'Phi2', 'theta2', 'psi2']].isna().any().any():
        problems.append('NaN in a required column')
    if problems:
        print()
        for p in problems:
            print('  FAIL: %s' % p)
        raise SystemExit(1)

    if args.dry_run:
        print('\n  --dry-run: not written')
        return df
    path = save_assembly_csv(df, name=args.name)
    print('\n  wrote %s' % path)
    return df


if __name__ == '__main__':
    main()
