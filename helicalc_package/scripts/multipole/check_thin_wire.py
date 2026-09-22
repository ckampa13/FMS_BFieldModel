'''
First-line regression on the multipole geometry -- CPU only, seconds to run.

Checks the thin-wire (bar-centreline) field, which needs no GPU and is exact in
the thin-wire limit.  Run this after ANY change to helicalc/multipole.py, before
spending GPU time.

    python check_thin_wire.py
    python check_thin_wire.py --aperture 0.100 --winding-radius auto
'''
import argparse
import numpy as np
import pandas as pd

from helicalc.multipole import (
    MultipoleGeom, build_assembly, ASSEMBLY_ELEMENTS, MQ_HARMONIC_TARGETS,
    check_closure, min_conductor_radius, thin_wire_field, harmonics_from_field,
    bars_per_pole, bar_angles_grid, current_for_strength, multipole_coeffs_2d,
    discrete_sector_harmonics, solve_sector_blocks_discrete, bar_angles,
)
from helicalc import helicalc_dir

FAILURES = []


def report(label, value, target, tol, unit='', fmt='%.5f'):
    ok = abs(value - target) <= tol
    flag = 'ok  ' if ok else 'FAIL'
    print(('  [%s] %-34s ' + fmt + ' %s (target ' + fmt + ' +/- ' + fmt + ')')
          % (flag, label, value, unit, target, tol))
    if not ok:
        FAILURES.append(label)
    return ok


def circle_harmonics(df, R, z, M=256, n_max=20):
    ph = 2 * np.pi * np.arange(M) / M
    pts = np.column_stack([R * np.cos(ph), R * np.sin(ph), np.full(M, z)])
    B = thin_wire_field(df, pts)
    return harmonics_from_field(B[:, 0], B[:, 1], ph, n_max=n_max)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--aperture', type=float, default=0.150)
    p.add_argument('--winding-radius', default='0.090')
    p.add_argument('--N-target', type=int, default=32)
    args = p.parse_args(argv)
    a = None if str(args.winding_radius).lower() in ('auto', 'none', '') \
        else float(args.winding_radius)
    geom = MultipoleGeom(aperture=args.aperture, a=a, N_target=args.N_target)

    print('=' * 78)
    print('1. Euler-angle convention vs real Mu2e bars')
    print('=' * 78)
    # helicalc's own geometry is the reference: reconstructing the far endpoint
    # from (origin, length, Phi2, theta2) must reproduce the CSV
    ref = pd.read_csv(helicalc_dir + 'dev/params/Mu2e_Straight_Bars_V13.csv')
    worst = 0.0
    for cn in (46, 57, 12):
        r = ref.query('`cond N` == %d' % cn).iloc[0]
        th, Ph = np.radians(r.theta2), np.radians(r.Phi2)
        d = np.array([np.sin(th) * np.cos(Ph), np.sin(th) * np.sin(Ph), np.cos(th)])
        p0 = np.array([r.x0, r.y0, r.z0]) if np.isclose(r.I_flow, 0.) \
            else np.array([r.x1, r.y1, r.z0])
        p1 = p0 + r.length * d
        if np.isclose(r.I_flow, 0.):
            tgt = np.array([r.x1, r.y1])
        else:
            tgt = np.array([r.x0, r.y0])
        if not np.any(np.isnan(tgt)):
            worst = max(worst, float(np.abs(p1[:2] - tgt).max()))
        # z_mid is an independent check and always present
        worst = max(worst, abs(0.5 * (p0[2] + p1[2]) - r.z_mid))
    report('Mu2e bar endpoint reconstruction', 1e3 * worst, 0.0, 1.0, 'mm', '%.4f')

    # our own round trip: angles -> rotation -> endpoint
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    w = 0.0
    for _ in range(200):
        q0 = rng.normal(size=3)
        q1 = q0 + rng.normal(size=3)
        L, Ph, th, ps = bar_angles(q0, q1)
        R = Rotation.from_euler('zyz', np.array([Ph, th, ps])[::-1], degrees=True)
        w = max(w, float(np.linalg.norm(q0 + R.apply([0, 0, L]) - q1)))
    report('bar_angles round trip', w, 0.0, 1e-9, 'm', '%.3e')

    print()
    print('=' * 78)
    print('2. cos(n.theta) normalisation (2D infinite-bar limit)')
    print('=' * 78)
    for n in range(1, 7):
        N = bars_per_pole(n, geom.N_target)
        theta = bar_angles_grid(N)
        I0 = current_for_strength(n, 1.0, N, geom)
        c = multipole_coeffs_2d(theta, I0 * np.cos(n * theta), geom, orders=[n])
        report('n=%d  B_n(R_ref) for B_ref=1 T' % n, c[n][0], 1.0, 1e-9, 'T')

    print()
    print('=' * 78)
    print('3. Assembly geometry')
    print('=' * 78)
    df = build_assembly(geom=geom)
    ok, worst_I, _ = check_closure(df)
    scale = float(np.abs(df['I']).max())
    report('Kirchhoff residual / peak current', worst_I / scale, 0.0, 1e-12,
           '', '%.3e')
    rmin = min_conductor_radius(df)
    print('  [%s] %-34s %.1f mm (must exceed R_map = %.1f and r_clear = %.1f)'
          % ('ok  ' if rmin >= max(geom.R_map, geom.r_clear) - 1e-9 else 'FAIL',
             'min conductor radius', 1e3 * rmin, 1e3 * geom.R_map,
             1e3 * geom.r_clear))
    if rmin < max(geom.R_map, geom.r_clear) - 1e-9:
        FAILURES.append('min conductor radius')
    print('  %-40s %d' % ('total bars', len(df)))
    print('  %-40s %.1f m' % ('total conductor length', df['length'].sum()))

    print()
    print('=' * 78)
    print('4. Element strengths at the element centre (thin wire, with ends)')
    print('=' * 78)
    print('  %-6s %-4s %-6s %-11s %-11s %-8s' %
          ('elem', 'n', 'skew', 'target [T]', 'built [T]', 'ratio'))
    for spec in ASSEMBLY_ELEMENTS:
        n = spec['n']
        zc = 0.5 * (spec['z0'] + spec['z1'])
        sub = df[df.element == spec['name']]
        h = circle_harmonics(sub, geom.R_ref, zc)
        val = h[n][1] if spec['skew'] else h[n][0]
        ratio = val / spec['B_ref']
        flag = 'ok  ' if 0.9 < ratio < 1.1 else 'FAIL'
        if flag == 'FAIL':
            FAILURES.append('strength %s' % spec['name'])
        print('  [%s] %-6s %-4d %-6s %-11.4f %-11.4f %-8.4f'
              % (flag, spec['name'], n, spec['skew'], spec['B_ref'], val, ratio))

    print()
    print('=' * 78)
    print('5. Main quad allowed harmonics')
    print('=' * 78)
    N = bars_per_pole(2, geom.N_target)
    blocks = solve_sector_blocks_discrete(2, MQ_HARMONIC_TARGETS, N, geom)
    print('  solved blocks (deg): %s'
          % [(round(lo, 4), round(hi, 4)) for lo, hi in blocks])
    hd = discrete_sector_harmonics(2, blocks, N, geom, orders=(6, 10, 14))
    for m, tgt in sorted(MQ_HARMONIC_TARGETS.items()):
        report('b%d, analytic 2D' % m, hd[m], tgt, 1e-3, 'units', '%.5f')
    print('  %-40s %+.4f units (free residual)' % ('b14, analytic 2D', hd[14]))
    # and on the winding as actually built (ends included)
    mq = df[df.element == 'MQ']
    h = circle_harmonics(mq, geom.R_ref, 0.5 * (0.400 + 1.900), M=1024)
    b2 = h[2][0]
    print('  built MQ (ends included): b6=%+.4f b10=%+.4f b14=%+.4f units'
          % tuple(1e4 * h[m][0] / b2 for m in (6, 10, 14)))

    print()
    print('=' * 78)
    print('6. Spurious harmonics of a single cos(n.theta) winding')
    print('=' * 78)
    print('  discretisation should show up only at orders n +/- N')
    for n in (2, 4):
        from helicalc.multipole import make_cosn_winding
        w = make_cosn_winding(n, 20.0, 0.0, B_ref=1.0, geom=geom)
        h = circle_harmonics(w, geom.R_ref, 0.0, M=1024, n_max=24)
        main = h[n][0]
        spur = max(abs(h[m][0] / main) for m in range(1, 25)
                   if m != n and abs(m - n) % bars_per_pole(n, geom.N_target) != 0)
        report('n=%d spurious/main below order 24' % n, 1e4 * spur, 0.0, 1.0,
               'units', '%.5f')

    print()
    print('=' * 78)
    if FAILURES:
        print('FAILED: %s' % ', '.join(FAILURES))
        raise SystemExit(1)
    print('All thin-wire checks passed.')
    return df


if __name__ == '__main__':
    main()
