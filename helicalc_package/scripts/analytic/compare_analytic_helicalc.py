'''
Compare the analytic nested-dipole maps (models A, B) with the helicalc Part A0 maps.

    python compare_analytic_helicalc.py            # whatever analytic maps exist

Reports, per model: |dB| by Z bin (measurement and test grids), the body agreement (relative),
the field at the Z-range ends, and the B_r harmonics at r = 60 mm by Z (odd n = 1..21, normal
and skew) for helicalc and the model.  Read-only.
'''
import os
from datetime import datetime

import numpy as np
import pandas as pd

HDIR = '/home/ckampa/data/Bmaps/multipole/'
ADIR = HDIR + 'analytic/'
GEOM = 'Multipole_HLLHC_V1_saddle'
REG = {'meas': 'measurement_region_MCB_Z1p9to5p1_k3_phi64', 'test': 'map_region_MCB_Z1p9to5p1'}
H = {'meas': HDIR + GEOM + '.' + REG['meas'] + '.summed.pkl', 'test': HDIR + GEOM + '.' + REG['test'] + '.summed.pkl'}
B = ['Bx', 'By', 'Bz']
BINS = [1.9, 2.2, 2.3, 2.35, 2.45, 2.5, 2.6, 3.0, 4.0, 4.4, 4.5, 4.55, 4.65, 4.7, 4.8, 5.12]


def absdiff(a, h):
    return 1e4 * np.sqrt(sum((a[c].values - h[c].values) ** 2 for c in B))


def harm(df, hp='r060mm'):
    d = df[df.HP == hp].copy()
    d['Br'] = d.Bx * np.cos(d.phi) + d.By * np.sin(d.phi)
    d['k'] = np.rint(d.phi * 64 / (2 * np.pi)).astype(int)
    d = d.sort_values(['Z', 'k'])
    zs = np.unique(d.Z)
    br = d.Br.values.reshape(len(zs), 64)
    return zs, np.fft.rfft(br, axis=1) * 2 / 64       # normal b_n = -Im c_n, skew a_n = Re c_n (B_r convention)


def main():
    print('compare_analytic_helicalc %s' % datetime.now().strftime('%F %T'))
    h = {k: pd.read_pickle(v) for k, v in H.items()}
    hz, hc = harm(h['meas'])
    for M in ('A', 'B'):
        paths = {k: ADIR + 'Multipole_HLLHC_V1_analytic_%s.%s.pkl' % (M, REG[k]) for k in REG}
        if not all(os.path.exists(p) for p in paths.values()):
            print('\nmodel %s: maps not found, skipped' % M)
            continue
        a = {k: pd.read_pickle(p) for k, p in paths.items()}
        for k in a:
            assert a[k][['X', 'Y', 'Z']].equals(h[k][['X', 'Y', 'Z']]), k
        print('\n==== model %s vs helicalc A0 ====' % M)
        print('|dB| [G] by Z bin: rms / max (meas | test);  helicalc |B| rms [G]')
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            row = []
            for k in ('meas', 'test'):
                s = (h[k].Z.values >= lo - 1e-9) & (h[k].Z.values < hi - 1e-9)
                d = absdiff(a[k][s], h[k][s])
                row.append('%8.2f / %8.2f' % (np.sqrt((d ** 2).mean()), d.max()) if s.any() else ' ' * 19)
            s = (h['meas'].Z.values >= lo - 1e-9) & (h['meas'].Z.values < hi - 1e-9)
            bh = 1e4 * np.sqrt((h['meas'][s][B] ** 2).sum(1))
            print('  %.2f-%.2f  %s | %s   %8.0f' % (lo, hi, row[0], row[1], np.sqrt((bh ** 2).mean())))
        for k in ('meas', 'test'):
            d = absdiff(a[k], h[k])
            bm = 1e4 * np.sqrt((h[k][B] ** 2).sum(1)).max()
            print('  all %s: rms %.2f G, max %.2f G (max |B| %.0f G)' % (k, np.sqrt((d ** 2).mean()), d.max(), bm))
        s = (h['meas'].Z > 2.8) & (h['meas'].Z < 4.2)
        d = absdiff(a['meas'][s], h['meas'][s])
        bm = 1e4 * np.sqrt((h['meas'][s][B] ** 2).sum(1))
        print('  body 2.8-4.2 (meas): rms %.3f G, max %.3f G = %.1e relative (max |dB| / |B|)' % (
            np.sqrt((d ** 2).mean()), d.max(), (d / bm).max()))
        for zz in (1.9, 5.1):
            for lab, df in (('helicalc', h['meas']), ('model ' + M, a['meas'])):
                p = df[np.abs(df.Z - zz) < 1e-6]
                print('  z = %.1f %-9s Bx %.1f..%.1f  By %.1f..%.1f  Bz %.1f..%.1f G' % (
                    zz, lab, *[1e4 * v for c in B for v in (p[c].min(), p[c].max())]))
        az, ac = harm(a['meas'])
        print('B_r harmonics at r = 60 mm [G]: max over the Z window (helicalc | model %s); normal b_n, skew a_n' % M)
        wins = [('ends 2.3-2.5', (2.3, 2.5)), ('body 2.8-4.2', (2.8, 4.2)), ('ends 4.5-4.7', (4.5, 4.7)),
                ('outside 1.9-2.3', (1.9, 2.3))]
        print('   n   ' + ''.join('%-36s' % w for w, _ in wins))
        for n in range(1, 22, 2):
            cells = []
            for _, (lo, hi) in wins:
                s = (hz >= lo - 1e-9) & (hz <= hi + 1e-9)
                hb, ha = np.abs(hc[s, n].imag).max(), np.abs(hc[s, n].real).max()
                mb, ma = np.abs(ac[s, n].imag).max(), np.abs(ac[s, n].real).max()
                cells.append('b %7.2f|%7.2f a %7.2f|%7.2f ' % (1e4 * hb, 1e4 * mb, 1e4 * ha, 1e4 * ma))
            print('  %2d  %s' % (n, ''.join(cells)))
        even = [n for n in range(0, 33) if n % 2 == 0]
        print('  even n (0..32) max, helicalc %.2e G, model %.2e G' % (
            1e4 * np.abs(hc[:, even]).max(), 1e4 * np.abs(ac[:, even]).max()))


if __name__ == '__main__':
    main()
