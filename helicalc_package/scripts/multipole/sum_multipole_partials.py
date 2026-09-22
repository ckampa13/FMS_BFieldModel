'''Sum the per-GPU partial field pickles from calculate_multipole_grid.py.'''
import argparse
import glob
import os

import pandas as pd

from helicalc import helicalc_data

OUTDIR = os.path.join(helicalc_data, 'Bmaps', 'multipole', '')


def sum_partials(pattern, out_name=None):
    files = sorted(glob.glob(os.path.join(OUTDIR, pattern)))
    if not files:
        raise SystemExit('no files match %s' % os.path.join(OUTDIR, pattern))
    total = None
    bcols = None
    for f in files:
        df = pd.read_pickle(f)
        if total is None:
            total = df.copy()
            bcols = [c for c in df.columns if c.startswith('B')]
        else:
            if len(df) != len(total):
                raise SystemExit('row count mismatch in %s' % f)
            for c in bcols:
                total[c] = total[c].values + df[c].values
        print('  + %s (%d rows)' % (os.path.basename(f), len(df)))
    if out_name is None:
        out_name = os.path.basename(files[0]).replace('.GPU', '.SUMMED_GPU')
        out_name = out_name.split('.SUMMED_GPU')[0] + '.summed.pkl'
    path = os.path.join(OUTDIR, out_name)
    total.to_pickle(path)
    print('wrote %s (%d files summed)' % (path, len(files)))
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pattern', help="e.g. 'Multipole_HLLHC_V1.body_region.GPU*_of_4.pkl'")
    p.add_argument('-o', '--out', default=None)
    a = p.parse_args()
    sum_partials(a.pattern, a.out)
