'''Analytic corrector package (plan 3, docdb models 19_*): the nine corrector elements of helicalc's ASSEMBLY_ELEMENTS
(MQS ... MCSS; no MQ/MCB/PERT) as model B ideal cos(n theta) sheets on the helicalc grid of the corrector-only map
(Z 4.4-8.6, dense Z 6.67 mm x 64 phi; FMS task 2), plus optional per-magnet random harmonics and rigid transverse offsets.

Pieces (each written once to <outdir>/parts_CORR/, then summed by --combine):
    design          the nine sheets on the nominal axis
    rh              random harmonics per magnet: normal + skew sheets n = 1..20 (not the magnet's own main term) on the
                    magnet's radius and z range, sigma_n = 1 unit (n <= 6), 0.6^(n-6) units above; 1 unit = 1e-4 of the
                    magnet's main field at R_ref; seed --rh-seed. On the nominal axis (their own feed-down ~ d/R x 1 unit).
    disp            the nine design sheets displaced by --offset-mm in a random direction per magnet (seed --dir-seed;
                    the same directions for every offset). Exact: B(x - dx, y - dy, z) of the centred sheet.
    leads           one closed go/return current-lead loop per corrector (exact filament Biot-Savart): from the coil terminal at
                    r = 90 mm at the magnet's downstream end, radially to r = 200 mm, axially to z = 10 m and back, the two
                    conductors dphi = 10 mm / 90 mm apart; azimuth and current sign random per magnet (seed --lead-seed);
                    --lead-current [A]. Free space (the superferric yoke would shield the axial runs): an upper bound.
    dipole          the upstream nested dipoles MCBH + MCBV (model B, analytic_multipole.nested_dipoles, z 2.4-4.6) on the
                    same grid: the map's upstream end (4.4) then sits in the dipole body (~2 T), as a partial-string map would.
    python build_analytic_correctors.py --part design|rh|disp [--offset-mm 0.1] [--workers 12]
    python build_analytic_correctors.py --combine [--rh] [--offset-mm 0.1] [--dipole]      -> the map + validation
Run in the helicalc env; the module is loaded by path from this worktree (not pip-installed).
'''
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
MOD = os.path.join(HERE, '..', '..', 'helicalc', 'analytic_multipole.py')
_spec = importlib.util.spec_from_file_location('analytic_multipole', MOD)
am = importlib.util.module_from_spec(_spec)
sys.modules['analytic_multipole'] = am      # so the element classes pickle for the worker pool
_spec.loader.exec_module(am)
from helicalc.multipole import ASSEMBLY_ELEMENTS, MultipoleGeom

HDIR = '/home/ckampa/data/Bmaps/multipole/'
GEOM = 'Multipole_HLLHC_V1_saddle'
REG = {'meas': 'measurement_region_CORR_Z4p4to8p6_k3_phi64', 'test': 'map_region_CORR_Z4p4to8p6',
       'cartoff': 'cartoff_region_CORR_Z4p4to8p6_xy5mm_zmid', 'cartz5': 'cartz5_region_CORR_Z4p4to8p6_xy5mm_z5mm'}
SRC = {'meas': HDIR + GEOM + '.' + REG['meas'] + '.summed.pkl', 'test': HDIR + GEOM + '.' + REG['test'] + '.summed.pkl'}


def grid(reg):
    '''positions of a region: the helicalc grids (meas, test), or the generated Z-offset uniform test grid (cartoff; docdb 2026-09-30):
    X, Y on a 5 mm Cartesian grid with r <= 60 mm, Z on the 630 mid-planes between the 631 measurement planes (4.4 + (j + 1/2) 4.2/630),
    so no position is shared with the measurement grid and the volume is sampled uniformly (as the Mu2e CartVal test map).
    cartz5: the same XY grid on Z = 4.4025 + 0.005 j (840 planes): incommensurate with the 6.667 mm measurement pitch (never on a
    measurement plane; phases 1/8, 3/8, 5/8, 7/8 of the pitch), so the test does not sit only at the mid-planes (Cole 2026-09-30).'''
    if reg in SRC:
        return pd.read_pickle(SRC[reg])
    if reg in ('cartoff', 'cartz5'):
        xy = np.round(np.arange(-12, 13) * 0.005, 9)
        X, Y = np.meshgrid(xy, xy, indexing='ij')
        keep = X ** 2 + Y ** 2 <= 0.06 ** 2 + 1e-12
        X, Y = X[keep], Y[keep]
        Z = np.round(4.4 + (np.arange(630) + 0.5) * 4.2 / 630, 9) if reg == 'cartoff' else np.round(4.4025 + 0.005 * np.arange(840), 9)
        g = pd.DataFrame({'X': np.tile(X, len(Z)), 'Y': np.tile(Y, len(Z)), 'Z': np.repeat(Z, len(X))})
        for c in ('Bx', 'By', 'Bz'):
            g[c] = 0.0
        return g
    raise ValueError(reg)
HPART = HDIR + GEOM + '.measurement_region_%s_Z4p4to8p6_k3_phi64.GPU0_of_1.pkl'
UNION = [0, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 18]      # allowed orders N(2k+1) of the correctors present (+ 0)
N_MAX_RH = 20


def log(msg):
    print('[%s] %s' % (datetime.now().strftime('%F %T'), msg))
    sys.stdout.flush()


def sigma_units(n):
    return 1.0 if n <= 6 else 0.6 ** (n - 6)


# HL-LHC TDR (CERN-2020-010) Table A-11, MQXF random column R at top energy, r0 = 50 mm, units; n = 3..14 (n > 14: 0). n = 1, 2 are not
# in the table (b2 R there is the quadrupole's own strength error): the n = 3 values are used for them (assumption).
MQXF_R_B = {3: 0.82, 4: 0.57, 5: 0.42, 6: 1.10, 7: 0.19, 8: 0.13, 9: 0.07, 10: 0.20, 11: 0.026, 12: 0.018, 13: 0.009, 14: 0.023}
MQXF_R_A = {3: 0.65, 4: 0.65, 5: 0.43, 6: 0.31, 7: 0.19, 8: 0.11, 9: 0.08, 10: 0.04, 11: 0.026, 12: 0.014, 13: 0.010, 14: 0.005}


def sigma_spectrum(n, skew, spectrum):
    '''sigma [units] of the random b_n (skew=False) / a_n (skew=True) for the named spectrum'''
    if spectrum == 'placeholder':
        return sigma_units(n)
    if spectrum == '10x':                      # the corrector design level (TDR Sec. 3.4: harmonics ~10 units) at low n, same falloff
        return 10.0 * sigma_units(n)
    if spectrum == 'mqxf':
        t = MQXF_R_A if skew else MQXF_R_B
        return t.get(max(n, 3), 0.0)
    raise ValueError(spectrum)


def design_elements():
    g = MultipoleGeom(aperture=0.150, a=0.090)
    els = []
    for s in ASSEMBLY_ELEMENTS:
        if s['kind'] != 'corr':
            continue
        els.append(am.SheetMultipole(s['n'], s['skew'], s['B_ref'], s['z0'], s['z1'], g.a, R_ref=g.R_ref, name=s['name']))
    return els


def rh_elements(design, seed, spectrum='placeholder'):
    '''list (per magnet) of lists of random-harmonic sheets, and the drawn values [units]'''
    rng = np.random.default_rng(seed)
    groups, vals = [], {}
    for e in design:
        unit = 1e-4 * e.B_ref
        grp = []
        for n in range(1, N_MAX_RH + 1):
            for skew in (False, True):
                if n == e.n and skew == e.skew:
                    continue
                u = float(rng.normal(0.0, 1.0)) * sigma_spectrum(n, skew, spectrum)   # same draws for every spectrum
                lab = '%s_%s%d' % (e.name, 'a' if skew else 'b', n)
                vals[lab] = u
                grp.append(am.SheetMultipole(n, skew, u * unit, e.z0, e.z1, e.a, R_ref=e.R_ref, name='rh_' + lab,
                                             calibrate=False))
        groups.append(grp)
    return groups, vals


def lead_elements(design, I, seed, r_t=0.090, r_b=0.200, sep=0.010, z_far=10.0):
    rng = np.random.default_rng(seed)
    cyl = lambda r, p, z: (r * np.cos(p), r * np.sin(p), z)
    els, info = [], {}
    dp = sep / r_t
    for e in design:
        phi = float(rng.uniform(0, 2 * np.pi)); sgn = float(rng.choice([-1.0, 1.0])); z1 = e.z1
        C = [cyl(r_t, phi - dp / 2, z1), cyl(r_b, phi - dp / 2, z1), cyl(r_b, phi - dp / 2, z_far),
             cyl(r_b, phi + dp / 2, z_far), cyl(r_b, phi + dp / 2, z1), cyl(r_t, phi + dp / 2, z1)]
        els.append(am.Polyline(C, sgn * I, name='lead_' + e.name))
        info[e.name] = dict(phi_rad=phi, sign=sgn, z_exit=z1, I_A=sgn * I)
    return els, dict(r_terminal=r_t, r_busbar=r_b, sep_m=sep, z_far=z_far, seed=seed, leads=info)


def directions(design, seed):
    rng = np.random.default_rng(seed)
    return {e.name: float(rng.uniform(0, 2 * np.pi)) for e in design}


def _eval(args):
    pos, els = args
    o = am.add_field(pos, els)
    return o[['Bx', 'By', 'Bz']].values


def offset_tag(d_mm):
    return ('%g' % d_mm).replace('.', 'p') + 'mm'


def part_path(outdir, part, reg, d_mm=None, rh_seed=None, dir_seed=None, lead_current=None, lead_seed=None, rh_spectrum='placeholder'):
    if part == 'design':
        t = 'design'
    elif part == 'dipole':
        t = 'dipole_MCB'
    elif part == 'leads':
        t = 'leads_%dA_s%d' % (round(lead_current), lead_seed)
    elif part == 'rh':
        t = 'rh_s%d' % rh_seed if rh_spectrum == 'placeholder' else 'rh_%s_s%d' % (rh_spectrum, rh_seed)
    else:
        t = 'disp_%s_s%d' % (offset_tag(d_mm), dir_seed)
    return os.path.join(outdir, 'parts_CORR', 'CORR_%s.%s.pkl' % (t, REG[reg]))


def build_part(a):
    design = design_elements()
    if a.part == 'design':
        tasks_els = [[e] for e in design]
        info = {e.name: dict(n=e.n, skew=e.skew, B_ref=e.B_ref, z0=e.z0, z1=e.z1, a=e.a, calib_scale=e.scale) for e in design}
    elif a.part == 'leads':
        tasks_els, info = lead_elements(design, a.lead_current, a.lead_seed)
        tasks_els = [[e] for e in tasks_els]
    elif a.part == 'dipole':
        dip = am.nested_dipoles('B')
        tasks_els = [[e] for e in dip]
        info = {e.name: dict(n=e.n, skew=e.skew, B_ref=e.B_ref, z0=e.z0, z1=e.z1, a=e.a, calib_scale=e.scale) for e in dip}
    elif a.part == 'rh':
        tasks_els, vals = rh_elements(design, a.rh_seed, a.rh_spectrum)
        info = dict(seed=a.rh_seed, spectrum=a.rh_spectrum, sigma='placeholder: 1 unit n<=6, 0.6^(n-6) above, n 1..20; 10x: x10; mqxf: TDR Table A-11 R', values_units=vals)
    else:
        dirs = directions(design, a.dir_seed)
        d = 1e-3 * a.offset_mm
        tasks_els = [[am.Displaced(e, d * np.cos(dirs[e.name]), d * np.sin(dirs[e.name]))] for e in design]
        info = dict(offset_mm=a.offset_mm, dir_seed=a.dir_seed, directions_rad=dirs)
    log('part %s: %s' % (a.part, json.dumps(info)[:2000]))
    os.makedirs(os.path.join(a.outdir, 'parts_CORR'), exist_ok=True)
    for reg in a.regions:
        path = part_path(a.outdir, a.part, reg, a.offset_mm, a.rh_seed, a.dir_seed, a.lead_current, a.lead_seed, a.rh_spectrum)
        if os.path.exists(path):
            log('EXISTS, not overwritten: %s' % path)
            continue
        h = grid(reg)
        pos = h[['X', 'Y', 'Z']].copy()
        t1 = time.time()
        with Pool(a.workers) as pool:
            res = pool.map(_eval, [(pos, els) for els in tasks_els])
        B = np.sum(res, axis=0)
        if not np.isfinite(B).all():
            raise SystemExit('non-finite field values; %s NOT written' % path)
        out = h.drop(columns=['Bx', 'By', 'Bz']).copy()
        out['Bx'], out['By'], out['Bz'] = B[:, 0], B[:, 1], B[:, 2]
        out = out[list(h.columns)]
        out.to_pickle(path)
        log('wrote %s (%d rows, %.0f s)' % (path, len(out), time.time() - t1))
    json.dump(dict(part=a.part, info=info, regions=a.regions, when=datetime.now().isoformat()),
              open(part_path(a.outdir, a.part, a.regions[0], a.offset_mm, a.rh_seed, a.dir_seed, a.lead_current, a.lead_seed, a.rh_spectrum).replace('.pkl', '.json'), 'w'), indent=1)


def harmonics_r60(df):
    '''|c_n| of B_r over the 64 phi at each Z (n = 0..32), r = 60 mm'''
    d = df[df.HP == 'r060mm'].copy()
    d['Br'] = d.Bx * np.cos(d.phi) + d.By * np.sin(d.phi)
    d['k'] = np.rint(np.mod(d.phi, 2 * np.pi) * 64 / (2 * np.pi)).astype(int) % 64
    d = d.sort_values(['Z', 'k'])
    zs = np.unique(d.Z)
    br = d.Br.values.reshape(len(zs), 64)
    return zs, np.fft.rfft(br, axis=1) * 2 / 64


def combine(a):
    tag = 'analytic_B_CORR'
    if a.rh:
        tag += ('_RHs%d' % a.rh_seed) if a.rh_spectrum == 'placeholder' else ('_RH%ss%d' % (a.rh_spectrum, a.rh_seed))
    if a.offset_mm:
        tag += '_d%s_s%d' % (offset_tag(a.offset_mm), a.dir_seed)
    if a.dipole:
        tag += '_MCB'
    if a.leads:
        tag += '_L%dA_s%d' % (round(a.lead_current), a.lead_seed)
    outs = {}
    for reg in a.regions:
        path = os.path.join(a.outdir, 'Multipole_HLLHC_V1_%s.%s.pkl' % (tag, REG[reg]))
        base = part_path(a.outdir, 'disp' if a.offset_mm else 'design', reg, a.offset_mm, a.rh_seed, a.dir_seed)
        parts = [base] + ([part_path(a.outdir, 'rh', reg, rh_seed=a.rh_seed, rh_spectrum=a.rh_spectrum)] if a.rh else []) + \
            ([part_path(a.outdir, 'dipole', reg)] if a.dipole else []) + \
            ([part_path(a.outdir, 'leads', reg, lead_current=a.lead_current, lead_seed=a.lead_seed)] if a.leads else [])
        ds = [pd.read_pickle(p) for p in parts]
        for d in ds[1:]:
            assert d[['X', 'Y', 'Z']].equals(ds[0][['X', 'Y', 'Z']])
        out = ds[0].copy()
        for c in ('Bx', 'By', 'Bz'):
            out[c] = sum(d[c].values for d in ds)
        h = grid(reg)
        assert np.allclose(out[['X', 'Y', 'Z']].values, h[['X', 'Y', 'Z']].values, atol=1e-9)
        if os.path.exists(path):
            log('EXISTS, not overwritten: %s' % path)
        else:
            out.to_pickle(path)
            log('wrote %s (%d rows) = %s' % (path, len(out), ' + '.join(os.path.basename(p) for p in parts)))
        outs[reg] = (path, out)
    if 'meas' not in outs:
        for reg, (p_, o_) in outs.items():
            log('%s: max |B_i| %.5f T, %d rows' % (reg, o_[['Bx', 'By', 'Bz']].abs().max().max(), len(o_)))
        return
    # validation: harmonic content at r = 60 mm by order class, and max |B_i|
    m = outs['meas'][1]
    zs, C = harmonics_r60(m)
    A = np.abs(C) * 1e4
    out_union = [n for n in range(1, 32) if n not in UNION]
    log('max |B_i| meas %.5f T; test %.5f T' % (m[['Bx', 'By', 'Bz']].abs().max().max(),
                                                   outs['test'][1][['Bx', 'By', 'Bz']].abs().max().max()))
    log('harmonics at r = 60 mm, max over Z [G]: ' + ', '.join('n%d %.3g' % (n, A[:, n].max()) for n in range(1, 21)))
    log('  in-union (1..31) RMS over Z of the per-Z quadrature sum: %.3f G; out-of-union: %.3f G (orders %s)' % (
        np.sqrt(np.mean(np.sum(A[:, [n for n in UNION if n > 0]] ** 2, axis=1))),
        np.sqrt(np.mean(np.sum(A[:, out_union] ** 2, axis=1))), out_union[:10]))
    # per-element body strength at the element centre (main term) and the n = 1 at the MQS centre, r = 60 mm scaled to R_ref
    for e in design_elements():
        zc = 0.5 * (e.z0 + e.z1)
        i = np.argmin(np.abs(zs - zc))
        c = C[i]
        main = (-c.imag[e.n] if not e.skew else c.real[e.n]) * (0.05 / 0.06) ** (e.n - 1)
        log('  %-5s z %.3f: main %s%d at R_ref %.4f T (design %.2f); |c_1| at r 60 %.3f G' % (
            e.name, zs[i], 'A' if e.skew else 'B', e.n, main, e.B_ref, 1e4 * np.abs(c[1])))
    json.dump(dict(tag=tag, outputs={k: v[0] for k, v in outs.items()}, when=datetime.now().isoformat()),
              open(os.path.join(a.outdir, 'Multipole_HLLHC_V1_%s.CORR_Z4p4to8p6.json' % tag), 'w'), indent=1)


def compare_helicalc(a):
    '''design part vs the helicalc corrector-only partials, per element, meas grid [G]'''
    dsg = design_elements()
    h = pd.read_pickle(SRC['meas'])
    pos = h[['X', 'Y', 'Z']].copy()
    for e in dsg:
        hp = pd.read_pickle(HPART % e.name)
        assert hp[['X', 'Y', 'Z']].equals(h[['X', 'Y', 'Z']])
        B = _eval((pos, [e]))
        d = 1e4 * np.sqrt(((B - hp[['Bx', 'By', 'Bz']].values) ** 2).sum(axis=1))
        z = h.Z.values
        body = (z > e.z0 + 0.3 * (e.z1 - e.z0)) & (z < e.z1 - 0.3 * (e.z1 - e.z0))
        ends = (np.abs(z - e.z0) < 0.05) | (np.abs(z - e.z1) < 0.05)
        far = (z < e.z0 - 0.3) | (z > e.z1 + 0.3)
        hB = 1e4 * np.sqrt((hp[['Bx', 'By', 'Bz']].values ** 2).sum(axis=1))
        log('  %-5s |dB| rms/max [G]: body %.2f/%.2f (|B| max %.0f), ends +-5 cm %.1f/%.1f, far (> 0.3 m) %.3f/%.3f' % (
            e.name, np.sqrt(np.mean(d[body] ** 2)), d[body].max(), hB[body].max(),
            np.sqrt(np.mean(d[ends] ** 2)), d[ends].max(), np.sqrt(np.mean(d[far] ** 2)), d[far].max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', choices=['design', 'rh', 'disp', 'dipole', 'leads'])
    ap.add_argument('--combine', action='store_true')
    ap.add_argument('--compare-helicalc', action='store_true')
    ap.add_argument('--rh', action='store_true', help='--combine: include the random harmonics')
    ap.add_argument('--dipole', action='store_true', help='--combine: include the upstream nested dipoles')
    ap.add_argument('--leads', action='store_true', help='--combine: include the current leads')
    ap.add_argument('--lead-current', type=float, default=200.0)
    ap.add_argument('--lead-seed', type=int, default=48753)
    ap.add_argument('--offset-mm', type=float, default=0.0)
    ap.add_argument('--rh-seed', type=int, default=48751)
    ap.add_argument('--rh-spectrum', default='placeholder', choices=['placeholder', 'mqxf', '10x'])
    ap.add_argument('--dir-seed', type=int, default=48752)
    ap.add_argument('--workers', type=int, default=9)
    ap.add_argument('--outdir', default=HDIR + 'analytic/')
    ap.add_argument('--regions', default='meas,test', help="comma list of %s (cartoff: Z-offset uniform test grid)" % list(REG))
    a = ap.parse_args()
    a.regions = a.regions.split(',')
    assert all(r in REG for r in a.regions), a.regions
    t0 = time.time()
    if a.part:
        if a.part == 'disp' and not a.offset_mm:
            raise SystemExit('--part disp needs --offset-mm')
        build_part(a)
    if a.compare_helicalc:
        compare_helicalc(a)
    if a.combine:
        combine(a)
    log('done, %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
