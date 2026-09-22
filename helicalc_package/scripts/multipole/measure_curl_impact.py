'''
Measure the FIELD-LEVEL impact of the prism-corner curl artefact.

The question this answers: helicalc's 3D bar field has curl B ~ 0.1 T/m in the
bore where it should be zero (see helicalc.multipole_field.add_field.__doc__).
A scalar-potential model -- which is what the paper's PINN is -- is curl-free by
construction, so it cannot represent that part of the field at all.  How many
gauss is it?

Method.  In a small ball inside the mapping volume, fit

    B = grad(phi),   phi = polynomial of degree <= D

by least squares.  That model space is exactly the curl-free fields, so whatever
is left over is the part no scalar potential can reproduce.  The same fit is run
on the thin-wire field of the identical geometry, which is exactly curl-free, so
its residual measures only polynomial truncation and sampling -- the control.
The 3D residual minus that control is the artefact, in gauss, to compare against
the 0.3 G Hall-probe noise.

    python measure_curl_impact.py
    python measure_curl_impact.py --radius 0.02 --degree 7
'''
import argparse
import itertools

import numpy as np
import pandas as pd

from helicalc.multipole import (MultipoleGeom, load_assembly_csv, thin_wire_field,
                                ASSEMBLY_ELEMENTS)
from helicalc.multipole_field import add_field


def grad_monomial_basis(P, degree, scale):
    '''Design matrix for B = grad(phi) with phi a polynomial of degree <= `degree`.

    Columns are the gradients of each monomial (x/s)^a (y/s)^b (z/s)^c with
    1 <= a+b+c <= degree, evaluated at the points P (centred already).  Spanning
    the gradients of all polynomials spans exactly the curl-free fields
    representable at this order, so the least-squares residual is the part of B
    that no scalar potential can produce.
    '''
    x, y, z = (P / scale).T
    cols = []
    for d in range(1, degree + 1):
        for a, b, c in itertools.product(range(d + 1), repeat=3):
            if a + b + c != d:
                continue
            # d/dx of x^a y^b z^c, etc.
            gx = (a * x ** (a - 1) * y ** b * z ** c) if a else np.zeros_like(x)
            gy = (b * x ** a * y ** (b - 1) * z ** c) if b else np.zeros_like(x)
            gz = (c * x ** a * y ** b * z ** (c - 1)) if c else np.zeros_like(x)
            cols.append(np.concatenate([gx, gy, gz]) / scale)
    return np.array(cols).T


def fit_curl_free(P, B, degree, scale):
    '''Least-squares fit of a curl-free field; returns (residual_vectors, rank).'''
    A = grad_monomial_basis(P, degree, scale)
    b = np.concatenate([B[:, 0], B[:, 1], B[:, 2]])
    coef, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ coef
    n = len(P)
    return np.array([res[:n], res[n:2 * n], res[2 * n:]]).T, rank


def probe_ball(center, radius, n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    r = radius * rng.random(n) ** (1 / 3)
    return np.asarray(center, float) + v * r[:, None]


def z_scan(bars, geom, args):
    """Residual vs z, to show which part of the map the artefact contaminates."""
    from helicalc.multipole import MAPPING_VOLUME
    z0 = MAPPING_VOLUME['z0'] + args.radius + 0.01
    z1 = MAPPING_VOLUME['z1'] - args.radius - 0.01
    zs = np.arange(z0, z1, args.z_step)
    rows = []
    for z in zs:
        c = np.array([0.030, 0.0, z])
        P = probe_ball(c, args.radius, args.npts, seed=int(z * 1000))
        df = pd.DataFrame({'X': P[:, 0], 'Y': P[:, 1], 'Z': P[:, 2]})
        B3 = add_field(df, bars, dev=args.Device, verbose=False,
                       tqdm=None)[['Bx', 'By', 'Bz']].values
        Bw = thin_wire_field(bars, P)
        r3, _ = fit_curl_free(P - c, B3, args.degree, args.radius)
        rw, _ = fit_curl_free(P - c, Bw, args.degree, args.radius)
        rows.append(dict(z=z,
                         B_G=1e4 * np.linalg.norm(B3, axis=1).mean(),
                         rms3_G=1e4 * np.sqrt((r3 ** 2).sum(axis=1).mean()),
                         rms_thin_G=1e4 * np.sqrt((rw ** 2).sum(axis=1).mean())))
    out = pd.DataFrame(rows)
    ends = []
    for spec in ASSEMBLY_ELEMENTS:
        ends += [spec['z0'], spec['z1']]
    ends = np.array(ends)
    out['d_to_end'] = [np.abs(ends - z).min() for z in out.z]

    over = out[out.rms3_G > 0.3]
    print('curl-free fit residual along z (ball r=%.0f mm at x=30 mm, degree %d)'
          % (1e3 * args.radius, args.degree))
    print()
    print('  %-8s %-10s %-12s %-12s %-10s' % ('z [m]', '|B| [G]', 'rms [G]',
                                              'control [G]', 'd_to_end'))
    for _, r in out.iterrows():
        mark = ' *' if r.rms3_G > 0.3 else ''
        print('  %-8.2f %-10.1f %-12.4f %-12.5f %-10.3f%s'
              % (r.z, r.B_G, r.rms3_G, r.rms_thin_G, r.d_to_end, mark))
    print()
    print('  %d of %d sample points exceed 0.3 G (%.0f%% of the z range)'
          % (len(over), len(out), 100.0 * len(over) / len(out)))
    if len(over):
        print('  worst %.2f G at z=%.2f m; all offenders lie within %.2f m of an'
              ' element end' % (over.rms3_G.max(), over.loc[over.rms3_G.idxmax(), 'z'],
                                over.d_to_end.max()))
    print('  max control residual %.2e G (basis is complete)' % out.rms_thin_G.max())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-g', '--Geom', default='Multipole_HLLHC_V1')
    p.add_argument('-D', '--Device', type=int, default=0)
    # defaults chosen from a convergence scan: at radius 10 mm / degree 8 the
    # thin-wire control fits to 3e-3 G, far below the 0.3 G noise, so the basis
    # is complete and the 3D residual is physical rather than truncation
    p.add_argument('--radius', type=float, default=0.010, help='probe ball radius [m]')
    p.add_argument('--degree', type=int, default=8, help='max polynomial degree')
    p.add_argument('--npts', type=int, default=600)
    p.add_argument('--aperture', type=float, default=0.150)
    p.add_argument('--winding-radius', default='0.090')
    p.add_argument('--z-scan', action='store_true',
                   help='profile the residual along z instead of at fixed sites')
    p.add_argument('--z-step', type=float, default=0.1)
    args = p.parse_args(argv)
    a = None if str(args.winding_radius).lower() in ('auto', 'none', '') \
        else float(args.winding_radius)
    geom = MultipoleGeom(aperture=args.aperture, a=a)
    bars = load_assembly_csv(args.Geom)

    # probe where the artefact was localised (element ends) and, as a contrast,
    # in the middle of the longest magnets where the two ends' effects cancel
    sites = [
        ('MQ end  (z=0.38)',   (0.030, 0.000, 0.381)),
        ('MQ centre (z=1.15)', (0.030, 0.000, 1.150)),
        ('MCBH end (z=4.61)',  (0.030, 0.000, 4.606)),
        ('MCBV mid (z=3.50)',  (0.030, 0.000, 3.500)),
        ('MCS end (z=7.85)',   (0.030, 0.000, 7.854)),
        ('quiet gap (z=6.31)', (0.030, 0.000, 6.313)),
    ]

    if args.z_scan:
        return z_scan(bars, geom, args)

    print('curl-free fit residual inside a ball of radius %.0f mm, '
          'phi degree <= %d' % (1e3 * args.radius, args.degree))
    print('thin wire is the control: it is exactly curl-free, so its residual')
    print('is pure truncation.  The 3D excess over it is the artefact.')
    print()
    print('  %-20s %-9s %-11s %-11s %-11s %-11s'
          % ('site', '|B| [G]', '3D rms [G]', '3D max [G]',
             'thin rms [G]', 'excess [G]'))
    rows = []
    for label, c in sites:
        P = probe_ball(c, args.radius, args.npts)
        df = pd.DataFrame({'X': P[:, 0], 'Y': P[:, 1], 'Z': P[:, 2]})
        B3 = add_field(df, bars, dev=args.Device, verbose=False,
                       tqdm=None)[['Bx', 'By', 'Bz']].values
        Bw = thin_wire_field(bars, P)
        r3, rank = fit_curl_free(P - c, B3, args.degree, args.radius)
        rw, _ = fit_curl_free(P - c, Bw, args.degree, args.radius)
        rms3 = 1e4 * np.sqrt((r3 ** 2).sum(axis=1).mean())
        max3 = 1e4 * np.linalg.norm(r3, axis=1).max()
        rmsw = 1e4 * np.sqrt((rw ** 2).sum(axis=1).mean())
        mag = 1e4 * np.linalg.norm(B3, axis=1).mean()
        flag = ('  <-- control not converged'
                if (rms3 > 1e-2 and rmsw > 0.05 * rms3) else '')
        print('  %-20s %-9.1f %-11.4f %-11.4f %-11.4f %-11.4f%s'
              % (label, mag, rms3, max3, rmsw, max(rms3 - rmsw, 0.0), flag))
        rows.append(dict(site=label, B_G=mag, rms3_G=rms3, max3_G=max3,
                         rms_thin_G=rmsw, rank=rank))
    print()
    print('  Hall-probe noise in the paper is sigma = 0.3 G per component.')
    print('  The thin-wire column must be far below the 3D column for the')
    print('  measurement to mean anything; if it is not, raise --degree or')
    print('  lower --radius until it is.')
    bad = [r for r in rows if r['rms3_G'] > 0.3]
    if bad:
        print()
        print('  %d of %d sites exceed the noise level, worst %.2f G rms at %s.'
              % (len(bad), len(rows), max(r['rms3_G'] for r in bad),
                 max(bad, key=lambda r: r['rms3_G'])['site']))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
