'''
Analytic nested-dipole (MCBH + MCBV) truth maps on the Part A0 grids, with validation.

    python build_analytic_dipole.py --model B
    python build_analytic_dipole.py --model A --enge-deg 3
    python build_analytic_dipole.py --model B --rand-units 1 --rand-orders 2-6 --seed 48750

--rand-units S adds "random" harmonics (model B only): for each n in --rand-orders a normal and a skew
ideal sheet on the MCBH winding over the same z range, strengths drawn N(0, S) units, 1 unit = 1e-4 of
B_ref (2.05 T) at R_ref.  The tag becomes analytic_B_rand<S>u_n<orders>_s<seed>; the drawn values go in
the log and the JSON sidecar.

Positions are copied from the helicalc A0 files, so the analytic and helicalc maps share every
point.  Output (tesla, same columns as helicalc) under /home/ckampa/data/Bmaps/multipole/analytic/:
    Multipole_HLLHC_V1_analytic_<M>.measurement_region_MCB_Z1p9to5p1_k3_phi64.pkl
    Multipole_HLLHC_V1_analytic_<M>.map_region_MCB_Z1p9to5p1.pkl
Existing files are never overwritten.  Model A's Enge coefficients are fitted per coil to the
helicalc on-axis field (A0 partials), where it is above 1 % of B_ref.

Run in the helicalc env.  The module is loaded by path from this worktree (not pip-installed).
'''
import argparse
import glob
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.realpath(__file__))
MOD = os.path.join(HERE, '..', '..', 'helicalc', 'analytic_multipole.py')
_spec = importlib.util.spec_from_file_location('analytic_multipole', MOD)
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)

HDIR = '/home/ckampa/data/Bmaps/multipole/'
GEOM = 'Multipole_HLLHC_V1_saddle'
MEAS = HDIR + GEOM + '.measurement_region_MCB_Z1p9to5p1_k3_phi64.summed.pkl'
TEST = HDIR + GEOM + '.map_region_MCB_Z1p9to5p1.summed.pkl'
PART = HDIR + GEOM + '.measurement_region_%s_Z1p9to5p1_k3_phi64.GPU[01]_of_2.pkl'


def log(msg):
    print('[%s] %s' % (datetime.now().strftime('%F %T'), msg))
    sys.stdout.flush()


def helicalc_axis(name, comp):
    '''on-axis (Z, B_comp) of one coil from the two A0 partials'''
    fs = sorted(glob.glob(PART % name))
    d = [pd.read_pickle(f) for f in fs]
    ax = d[0].HP.values == 'r000mm'
    return d[0].Z.values[ax], (d[0][comp].values + d[1][comp].values)[ax]


def fit_enge(z, b, B_ref, z0, z1, D, deg, frac=0.01):
    '''Enge coefficients (a0..a_deg) of S so that B_ref S fits b where b > frac B_ref'''
    use = b > frac * B_ref

    def model(p, zz):
        E = lambda s: 1.0 / (1.0 + np.exp(np.clip(np.polyval(p[::-1], s), -700, 700)))
        return B_ref * E((zz - z1) / D) * E((z0 - zz) / D)
    p0 = np.zeros(deg + 1)
    p0[1] = 4.0
    f = least_squares(lambda p: model(p, z[use]) - b[use], p0)
    res = model(f.x, z[use]) - b[use]
    # nearest complex pole of E(s): p(s) = i pi (2j + 1)
    dmin = np.inf
    for j in range(-4, 4):
        c = f.x.astype(complex)
        c[0] -= 1j * np.pi * (2 * j + 1)
        for rt in np.roots(c[::-1]):
            if abs(rt.real) < 3:
                dmin = min(dmin, abs(rt.imag) * D)
    return f.x, np.sqrt(np.mean(res ** 2)), np.abs(res).max(), dmin


def random_harmonics(orders, units, seed, host):
    '''normal + skew ideal sheets of each order n on the host coil's radius and z range'''
    rng = np.random.default_rng(seed)
    unit = 1e-4 * host.B_ref
    els, vals = [], {}
    for n in orders:
        for skew in (False, True):
            u = float(rng.normal(0.0, units))
            lab = '%s%d' % ('a' if skew else 'b', n)
            vals[lab] = u
            els.append(am.SheetMultipole(n, skew, u * unit, host.z0, host.z1, host.a, R_ref=host.R_ref,
                                         name='rand_' + lab))
    return els, vals


def build_elements(model, deg, mmax, rand=None):
    info = {}
    if model == 'B':
        els = am.nested_dipoles('B')
        for e in els:
            info[e.name] = dict(a=e.a, z0=e.z0, z1=e.z1, calib_scale=e.scale, dk=e.dk, kcut=e.kcut)
        if rand is not None:
            extra, vals = random_harmonics(rand['orders'], rand['units'], rand['seed'], els[0])
            els += extra
            info['random_harmonics'] = dict(units_sigma=rand['units'], seed=rand['seed'], host='MCBH',
                                            unit_T_at_Rref=1e-4 * els[0].B_ref, values_units=vals)
        return els, info
    engs = {}
    for name, comp in (('MCBH', 'By'), ('MCBV', 'Bx')):
        z, b = helicalc_axis(name, comp)
        p, rms, mx, dmin = fit_enge(z, b, 2.05, 2.4, 4.6, 0.15, deg)
        engs[name] = p
        info[name] = dict(enge=[float(v) for v in p], fit_rms_G=1e4 * rms, fit_max_G=1e4 * mx,
                          nearest_pole_m=dmin)
    els = am.nested_dipoles('A', enge_H=tuple(engs['MCBH']), enge_V=tuple(engs['MCBV']))
    for e in els:
        info[e.name]['calib_scale'] = e.scale
        info[e.name]['mmax'] = mmax
    return els, info


def div_curl(els, n_pts=300, h=2.5e-4, seed=7, qkw=None):
    '''max |div B|, |curl B| [T/m] by 4th-order central differences at random bore points'''
    qkw = qkw or {}
    rng = np.random.default_rng(seed)
    r = 0.06 * np.sqrt(rng.random(n_pts))
    p = 2 * np.pi * rng.random(n_pts)
    # a third of the points within 0.1 m of a magnet end
    z = np.where(rng.random(n_pts) < 1 / 3.,
                 rng.choice([2.4, 4.6], n_pts) + rng.uniform(-0.1, 0.1, n_pts),
                 rng.uniform(1.9, 5.1, n_pts))
    x0, y0 = r * np.cos(p), r * np.sin(p)
    offs = [(-2, 1.0), (-1, -8.0), (1, 8.0), (2, -1.0)]    # f' = sum w f(x + s h) / (12 h)
    X, Y, Z = [], [], []
    for ax in range(3):
        for s, _ in offs:
            d = np.zeros(3)
            d[ax] = s * h
            X.append(x0 + d[0]); Y.append(y0 + d[1]); Z.append(z + d[2])
    X, Y, Z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)
    Bx, By, Bz = am.field_xyz(els, X, Y, Z, r_round=12, **qkw)
    B = np.stack([Bx, By, Bz]).reshape(3, 3, len(offs), n_pts)   # comp, axis, offset, point
    w = np.array([wt for _, wt in offs])[None, None, :, None]
    # the weights above are for f(x-2h), f(x-h), f(x+h), f(x+2h): (f(-2) - 8 f(-1) + 8 f(1) - f(2)) / 12h
    J = (B * w).sum(axis=2) / (12 * h)                             # J[comp, axis] = dB_comp / d axis
    div = J[0, 0] + J[1, 1] + J[2, 2]
    curl = np.sqrt((J[2, 1] - J[1, 2]) ** 2 + (J[0, 2] - J[2, 0]) ** 2 + (J[1, 0] - J[0, 1]) ** 2)
    ends = np.minimum(np.abs(z - 2.4), np.abs(z - 4.6)) < 0.1
    return div, curl, ends, r


def harmonics(df, r_hp='r060mm'):
    '''|c_n| of B_r over the 64 phi at each Z (n = 0..32), from a measurement-grid frame'''
    d = df[df.HP == r_hp].copy()
    d['Br'] = d.Bx * np.cos(d.phi) + d.By * np.sin(d.phi)
    d['k'] = np.rint(d.phi * 64 / (2 * np.pi)).astype(int)
    zs = np.unique(d.Z)
    C = []
    for z in zs:
        p = d[np.abs(d.Z - z) < 1e-9].sort_values('k')
        C.append(np.fft.rfft(p.Br.values) * 2 / 64)
    return zs, np.array(C)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=['A', 'B'])
    ap.add_argument('--enge-deg', type=int, default=3, help='model A: Enge polynomial degree (default 3)')
    ap.add_argument('--mmax', type=int, default=60, help='model A: highest series order m (default 60)')
    ap.add_argument('--outdir', default=HDIR + 'analytic/')
    ap.add_argument('--no-maps', action='store_true', help='validate only')
    ap.add_argument('--skip-divcurl', action='store_true', help='skip the (slow) div/curl check')
    ap.add_argument('--rand-units', type=float, default=0.0,
                    help='model B: sigma of random harmonics in units (1e-4 of B_ref at R_ref); 0 = none')
    ap.add_argument('--rand-orders', default='2-6', help='orders for --rand-units, e.g. 2-6 or 2,3,4')
    ap.add_argument('--seed', type=int, default=48750)
    args = ap.parse_args(argv)
    qkw = {'mmax': args.mmax} if args.model == 'A' else {}
    tag = 'analytic_%s' % args.model
    rand = None
    if args.rand_units > 0:
        if args.model != 'B':
            raise SystemExit('--rand-units needs --model B')
        if '-' in args.rand_orders:
            lo, hi = (int(v) for v in args.rand_orders.split('-'))
            orders = list(range(lo, hi + 1))
        else:
            orders = [int(v) for v in args.rand_orders.split(',')]
        rand = dict(units=args.rand_units, orders=orders, seed=args.seed)
        tag += '_rand%gu_n%s_s%d' % (args.rand_units, args.rand_orders.replace(',', '.'), args.seed)
    t0 = time.time()
    log('model %s (%s): building elements' % (args.model, tag))
    els, info = build_elements(args.model, args.enge_deg, args.mmax, rand)
    log('elements: %s' % json.dumps(info, indent=1))

    # ---- validation 1: div / curl
    if not args.skip_divcurl:
        div, curl, ends, rr = div_curl(els, qkw=qkw)
        log('(1) div/curl, 4th-order FD, h = 0.25 mm, 300 points r <= 60 mm (%d within 0.1 m of an end):' % ends.sum())
        for lab, s in (('all', np.ones_like(ends)), ('ends', ends), ('body', ~ends)):
            log('    %-4s max |div B| %.2e G/m, max |curl B| %.2e G/m'
                % (lab, 1e4 * np.abs(div[s]).max(), 1e4 * curl[s].max()))

    # ---- validation 2: body strength at R_ref, z = 3.5 (64 phi)
    ph = 2 * np.pi * np.arange(64) / 64
    for e in els:
        Bx, By, Bz = am.field_xyz([e], 0.05 * np.cos(ph), 0.05 * np.sin(ph), np.full(64, 3.5), **qkw)
        Br = Bx * np.cos(ph) + By * np.sin(ph)
        c = np.fft.rfft(Br) * 2 / 64
        # normal: B_r = B_n sin(n phi) -> -Im c_n; skew: B_r = A_n cos(n phi) -> Re c_n
        bn = -c.imag[e.n] if not e.skew else c.real[e.n]
        other = np.delete(np.abs(c), e.n).max()
        log('(2) %s body z = 3.5, r = R_ref: %s_%d = %.6f T (design %.2f); other |c_n| max %.1e T'
            % (e.name, 'A' if e.skew else 'B', e.n, bn, e.B_ref, other))

    if rand is not None:
        Bx, By, Bz = am.field_xyz(els, 0.05 * np.cos(ph), 0.05 * np.sin(ph), np.full(64, 3.5), **qkw)
        c = np.fft.rfft(Bx * np.cos(ph) + By * np.sin(ph)) * 2 / 64
        unit = 1e-4 * 2.05
        for n in [1] + rand['orders']:
            log('(2r) total field, z = 3.5, R_ref: n = %d  b = %+.4f  a = %+.4f units (drawn b %s, a %s)' % (
                n, -c.imag[n] / unit, c.real[n] / unit,
                '%+.4f' % info['random_harmonics']['values_units'].get('b%d' % n, float('nan')),
                '%+.4f' % info['random_harmonics']['values_units'].get('a%d' % n, float('nan'))))
    if args.model == 'A':
        # ---- validation 4: series convergence at r = 60 mm
        zz = np.linspace(1.9, 5.1, 641)
        for e in els:
            terms = e.series_terms(0.06, zz, args.mmax)
            dQr = np.abs(terms[1])                               # field-level size of each term
            last = dQr[:, -1]
            need = np.array([np.argmax(np.r_[row < 1e-8, True]) for row in dQr])   # first m with |term| < 1e-8 T
            log('(4) %s series at r = 60 mm: |term m = %d| max %.1e T (z = %.3f); m needed for < 1e-8 T (1e-4 G): '
                'median %d, max %d (z = %.3f)' % (e.name, args.mmax, last.max(), zz[np.argmax(last)],
                                                  np.median(need), need.max(), zz[np.argmax(need)]))

    if args.no_maps:
        log('done (validation only), %.0f s' % (time.time() - t0))
        return
    os.makedirs(args.outdir, exist_ok=True)
    outs = {}
    for src, reg in ((MEAS, 'measurement_region_MCB_Z1p9to5p1_k3_phi64'), (TEST, 'map_region_MCB_Z1p9to5p1')):
        path = os.path.join(args.outdir, 'Multipole_HLLHC_V1_%s.%s.pkl' % (tag, reg))
        if os.path.exists(path):
            log('EXISTS, not overwritten: %s' % path)
            continue
        h = pd.read_pickle(src)
        pos = h.drop(columns=['Bx', 'By', 'Bz'])
        t1 = time.time()
        out = am.add_field(pos, els, **qkw)
        out = out[list(h.columns)]
        assert out[['X', 'Y', 'Z']].equals(h[['X', 'Y', 'Z']])
        if not np.isfinite(out[['Bx', 'By', 'Bz']].values).all():
            raise SystemExit('non-finite field values; %s NOT written' % path)
        out.to_pickle(path)
        outs[reg] = path
        log('wrote %s (%d rows, %.0f s)' % (path, len(out), time.time() - t1))

    # ---- validation 3: harmonic content at r = 60 mm (measurement map)
    if 'measurement_region_MCB_Z1p9to5p1_k3_phi64' in outs:
        m = pd.read_pickle(outs['measurement_region_MCB_Z1p9to5p1_k3_phi64'])
        zs, C = harmonics(m)
        present = [1] + (rand['orders'] if rand else [])
        other = np.abs(np.delete(C, present, axis=1)).max()
        log('(3) harmonic content, r = 60 mm, all %d Z: |c_1| max %.4f T; n in %s max |c_n| %s G; all other n max %.1e T'
            % (len(zs), np.abs(C[:, 1]).max(), present[1:],
               [round(1e4 * np.abs(C[:, n]).max(), 3) for n in present[1:]], other))
    json.dump(dict(model=args.model, info=info, outputs=outs, when=datetime.now().isoformat()),
              open(os.path.join(args.outdir, 'Multipole_HLLHC_V1_%s.MCB_Z1p9to5p1.json' % tag), 'w'), indent=1)
    log('done, %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
