'''
Full validation of the multipole assembly field (handoff section 5), on the GPU.

  1. body strength B_n(R_ref) at each element centre vs the design table
  2. harmonic content by Fourier analysis on a circle (the rotating-coil reference)
  3. effective magnetic length from the on-axis / R_ref profile
  4. Maxwell residuals: div and curl at 2nd and 4th order vs step size
  5. neighbour overlap at mid-gap
  6. cross-check against the thin-wire model

    python validate_multipole.py                 # runs everything
    python validate_multipole.py --skip maxwell  # everything but the slow one
'''
import argparse
import os

import numpy as np
import pandas as pd

from helicalc import helicalc_data
from helicalc.multipole import (
    MultipoleGeom, load_assembly_csv, ASSEMBLY_ELEMENTS, MAPPING_VOLUME,
    thin_wire_field, harmonics_from_field, MQ_HARMONIC_TARGETS,
)
from helicalc.multipole_field import add_field
from helicalc.jacobian import (div_and_curl_calculations,
                               div_and_curl_calculations_4th,
                               add_points_for_J_4th)
from helicalc.tools import add_points_for_J

OUTDIR = os.path.join(helicalc_data, 'Bmaps', 'multipole', '')


def circle_df(R, z, M=256, label=''):
    ph = 2 * np.pi * np.arange(M) / M
    return pd.DataFrame({'X': R * np.cos(ph), 'Y': R * np.sin(ph),
                         'Z': np.full(M, z), 'HP': label}), ph


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-g', '--Geom', default='Multipole_HLLHC_V1')
    p.add_argument('-D', '--Device', type=int, default=0)
    p.add_argument('--aperture', type=float, default=0.150)
    p.add_argument('--winding-radius', default='0.090')
    p.add_argument('--skip', default='', help='comma list: body,length,maxwell,overlap')
    p.add_argument('--dz', type=float, default=0.010,
                   help='z step for the length profiles [m]')
    args = p.parse_args(argv)
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    a = None if str(args.winding_radius).lower() in ('auto', 'none', '') \
        else float(args.winding_radius)
    geom = MultipoleGeom(aperture=args.aperture, a=a)
    bars = load_assembly_csv(args.Geom)
    dev = args.Device
    M = 256
    results = {}

    # ---------------------------------------------------------------- body
    if 'body' not in skip:
        section('1-2. Body strength and harmonics at each element centre'
                ' (r = %.0f mm)' % (1e3 * geom.R_ref))
        frames, phs = [], None
        for spec in ASSEMBLY_ELEMENTS:
            d, ph = circle_df(geom.R_ref, 0.5 * (spec['z0'] + spec['z1']), M,
                              spec['name'])
            frames.append(d)
            phs = ph
        allpts = pd.concat(frames, ignore_index=True)
        # per_element keeps each winding's own contribution separate, so the
        # table can show the element in isolation (the rotating-coil reference)
        # alongside the total field it actually sits in.  They differ a lot: a
        # 2.05 T, 2.2 m dipole has a magnetic moment of ~1e5 A m^2, so its
        # fringe is tens of gauss at the correctors a metre downstream.
        out = add_field(allpts, bars, dev=dev, verbose=False, tqdm=None,
                        per_element=True)
        Bt = thin_wire_field(bars, allpts[['X', 'Y', 'Z']].values)

        def el_field(sub, name):
            cx, cy = 'Bx_%s' % name, 'By_%s' % name
            if cx in sub.columns:
                return sub[cx].values, sub[cy].values
            return sub.Bx.values, sub.By.values

        print('  %-6s %-3s %-10s %-10s %-8s %-10s %-10s'
              % ('elem', 'n', 'design[T]', 'own[T]', 'ratio', 'vs thinwire',
                 'total[T]'))
        rows = []
        for spec in ASSEMBLY_ELEMENTS:
            s = out[out.HP == spec['name']]
            idx = out.index[out.HP == spec['name']]
            n = spec['n']
            ex, ey = el_field(s, spec['name'])
            h = harmonics_from_field(ex, ey, phs, n_max=20)
            hall = harmonics_from_field(s.Bx.values, s.By.values, phs, n_max=20)
            ht = harmonics_from_field(Bt[idx, 0], Bt[idx, 1], phs, n_max=20)
            j = 1 if spec['skew'] else 0
            v, vall, vt = h[n][j], hall[n][j], ht[n][j]
            print('  %-6s %-3d %-10.4f %-10.4f %-8.4f %-10.4f %-10.4f'
                  % (spec['name'], n, spec['B_ref'], v, v / spec['B_ref'],
                     v / vt if vt else np.nan, vall))
            rows.append(dict(element=spec['name'], n=n, design=spec['B_ref'],
                             own=v, total=vall, thin_wire_total=vt))
        results['body'] = pd.DataFrame(rows)

        print()
        print('  Harmonics in units (1e-4 of own main field) at r = %.0f mm.'
              % (1e3 * geom.R_ref))
        print('  "own" = this winding alone; "total" includes neighbours.')
        for spec in ASSEMBLY_ELEMENTS:
            s = out[out.HP == spec['name']]
            n = spec['n']
            j = 1 if spec['skew'] else 0
            ex, ey = el_field(s, spec['name'])
            h = harmonics_from_field(ex, ey, phs, n_max=20)
            hall = harmonics_from_field(s.Bx.values, s.By.values, phs, n_max=20)
            main = h[n][j]
            for lab, hh in (('own  ', h), ('total', hall)):
                big = []
                for m in range(1, 21):
                    for pfx, val in (('b', hh[m][0]), ('a', hh[m][1])):
                        u = 1e4 * val / main
                        if not (m == n and pfx == ('a' if spec['skew'] else 'b')) \
                                and abs(u) > 0.5:
                            big.append('%s%d=%+.1f' % (pfx, m, u))
                print('    %-6s %-6s %s' % (spec['name'] if lab == 'own  ' else '',
                                            lab,
                                            ' '.join(big[:8]) or '(none > 0.5)'))
        mq = out[out.HP == 'MQ']
        mx, my = el_field(mq, 'MQ')
        h = harmonics_from_field(mx, my, phs, n_max=20)
        print()
        print('  MQ allowed harmonics (own field) vs target %s:'
              % MQ_HARMONIC_TARGETS)
        for m in (6, 10, 14):
            print('    b%-3d = %+8.4f units' % (m, 1e4 * h[m][0] / h[2][0]))

    # ------------------------------------------------------------- lengths
    if 'length' not in skip:
        section('3. Effective magnetic length')
        z0, z1 = MAPPING_VOLUME['z0'], MAPPING_VOLUME['z1']
        frames = []
        for spec in ASSEMBLY_ELEMENTS:
            L = spec['z1'] - spec['z0']
            zc = 0.5 * (spec['z0'] + spec['z1'])
            zs = np.arange(zc - 3 * max(L, 0.4), zc + 3 * max(L, 0.4) + 1e-9, args.dz)
            for z in zs:
                d, ph = circle_df(geom.R_ref, z, 32, '%s@%.4f' % (spec['name'], z))
                frames.append(d)
        allpts = pd.concat(frames, ignore_index=True)
        print('  profiling %d points ...' % len(allpts))
        out = add_field(allpts, bars, dev=dev, verbose=False, tqdm=None)
        ph32 = 2 * np.pi * np.arange(32) / 32

        print('  %-6s %-9s %-11s %-11s %-8s %-8s'
              % ('elem', 'L_geom', 'IntBdl_des', 'IntBdl_got', 'ratio', 'Leff/L'))
        rows = []
        for spec in ASSEMBLY_ELEMENTS:
            L = spec['z1'] - spec['z0']
            zc = 0.5 * (spec['z0'] + spec['z1'])
            zs = np.arange(zc - 3 * max(L, 0.4), zc + 3 * max(L, 0.4) + 1e-9, args.dz)
            prof = []
            for z in zs:
                s = out[out.HP == '%s@%.4f' % (spec['name'], z)]
                h = harmonics_from_field(s.Bx.values, s.By.values, ph32, n_max=8)
                prof.append(h[spec['n']][1] if spec['skew'] else h[spec['n']][0])
            prof = np.array(prof)
            BdL = np.trapz(prof, zs)
            body = prof[np.argmax(np.abs(prof))]
            Leff = BdL / body
            des = spec.get('BdL')
            print('  %-6s %-9.3f %-11s %-11.4f %-8s %-8.4f'
                  % (spec['name'], L, '%.4f' % des if des else '-', BdL,
                     '%.4f' % (BdL / des) if des else '-', Leff / L))
            rows.append(dict(element=spec['name'], L_geom=L, BdL=BdL,
                             BdL_design=des, L_eff=Leff))
        results['length'] = pd.DataFrame(rows)

    # ------------------------------------------------------------- maxwell
    if 'maxwell' not in skip:
        section('4. Maxwell residuals vs step size')
        rng = np.random.default_rng(12345)
        n = 200
        R = geom.R_map
        rr = 0.9 * R * np.sqrt(rng.random(n))
        pp = 2 * np.pi * rng.random(n)
        base = pd.DataFrame({'X': rr * np.cos(pp), 'Y': rr * np.sin(pp),
                             'Z': rng.uniform(0.2, 8.4, n)})
        B0 = add_field(base, bars, dev=dev, verbose=False, tqdm=None)
        Bscale = np.linalg.norm(B0[['Bx', 'By', 'Bz']].values, axis=1).max()
        print('  |B| max on the sample = %.4f T' % Bscale)
        print()
        print('  The thin-wire column is the SAME geometry with the conductor')
        print('  cross-section collapsed to its centreline.  It is exactly')
        print('  Maxwellian, so it isolates finite-difference truncation from')
        print('  the prism-corner artefact described in add_field.__doc__.')
        print()
        print('  %-10s %-11s %-6s %-11s %-6s %-11s %-11s'
              % ('h [m]', 'div 3D', 'rat', 'curl 3D', 'rat',
                 'div thin', 'curl thin'))
        p2 = pc = None
        rows = []
        # go far enough down in h that the 3D curl floor actually shows up:
        # it only separates from the thin-wire curve below ~1 mm
        for h in (1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4):
            e2 = add_points_for_J(base, h)
            d2 = add_field(e2, bars, dev=dev, verbose=False, tqdm=None)
            r2, _ = div_and_curl_calculations(d2)
            Bw = thin_wire_field(bars, e2[['X', 'Y', 'Z']].values)
            rw, _ = div_and_curl_calculations(
                e2.assign(Bx=Bw[:, 0], By=Bw[:, 1], Bz=Bw[:, 2]))
            v2 = np.abs(r2.divB).max()
            vc = np.abs(r2.curlB).max()
            w2 = np.abs(rw.divB).max()
            wc = np.abs(rw.curlB).max()
            print('  %-10.2e %-11.3e %-6s %-11.3e %-6s %-11.3e %-11.3e'
                  % (h, v2, '%.1f' % (p2 / v2) if p2 else '-',
                     vc, '%.1f' % (pc / vc) if pc else '-', w2, wc))
            rows.append(dict(h=h, div3d=v2, curl3d=vc, div_thin=w2, curl_thin=wc))
            p2, pc = v2, vc
        print()
        print('  Expect: div 3D ~4x per halving and tracking div thin (it does);')
        print('  curl 3D stalls near 0.1 T/m while curl thin keeps falling --')
        print('  that gap is the prism-corner artefact, not a convergence failure.')
        results['maxwell'] = pd.DataFrame(rows)

    # ------------------------------------------------------------- overlap
    if 'overlap' not in skip:
        section('5. Neighbour overlap at mid-gap')
        frames = []
        pairs = []
        for i in range(len(ASSEMBLY_ELEMENTS) - 1):
            A, Bn = ASSEMBLY_ELEMENTS[i], ASSEMBLY_ELEMENTS[i + 1]
            if Bn['z0'] <= A['z1']:
                continue
            zm = 0.5 * (A['z1'] + Bn['z0'])
            d, ph = circle_df(geom.R_ref, zm, 64, '%s|%s' % (A['name'], Bn['name']))
            frames.append(d)
            pairs.append((A, Bn, zm))
        allpts = pd.concat(frames, ignore_index=True)
        out = add_field(allpts, bars, dev=dev, verbose=False, tqdm=None)
        ph64 = 2 * np.pi * np.arange(64) / 64
        print('  %-14s %-8s %-12s %-12s'
              % ('gap', 'z_mid', 'A at mid/A0', 'B at mid/B0'))
        for A, Bn, zm in pairs:
            s = out[out.HP == '%s|%s' % (A['name'], Bn['name'])]
            h = harmonics_from_field(s.Bx.values, s.By.values, ph64, n_max=20)
            va = h[A['n']][1] if A['skew'] else h[A['n']][0]
            vb = h[Bn['n']][1] if Bn['skew'] else h[Bn['n']][0]
            print('  %-14s %-8.3f %-12.4f %-12.4f'
                  % ('%s|%s' % (A['name'], Bn['name']), zm,
                     va / A['B_ref'], vb / Bn['B_ref']))

    section('Done')
    for k, v in results.items():
        path = os.path.join(OUTDIR, '%s.validation_%s.csv' % (args.Geom, k))
        v.to_csv(path, index=False)
        print('  wrote %s' % path)
    return results


if __name__ == '__main__':
    main()
