'''Tests for the v1010 fit function (v1008 + k=0 multipole terms) and FieldFitter method="linear".'''
import numpy as np
import pandas as pd
import pytest
from lmfit import Parameters
from mu2e.tools import fit_funcs_redux as ff
from mu2e.fieldfitter_redux2 import FieldFitter
from mu2e.cfg_defs import cfg_params, cfg_pickle

L, MS, NS, Z0, RREF = 4.0, 4, 4, 1.0, 0.05


def points(N=400, seed=7, n_axis=10):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, 0.06, N)
    r[:n_axis] = 0.0  # on-axis rows
    phi = rng.uniform(-np.pi, np.pi, N)
    z = rng.uniform(0.0, 2.0, N)
    return z, r, phi, r*np.cos(phi), r*np.sin(phi)


def base_params(rng, ms=MS, n_list=range(NS), n_list_k0=(), ks=None):
    p = {'z0': Z0}
    for m in range(ms):
        for n in n_list:
            for P in 'ABCD':
                p[f'{P}c1_{m}_{n}'] = rng.normal() if not (n == 0 and P in 'CD') else 0.
    for n in n_list_k0:
        p[f'Ek0_{n}'] = rng.normal()
        p[f'Fk0_{n}'] = rng.normal()
    for i in range(1, 11):
        p[f'k{i}'] = 0. if ks is None else ks.get(f'k{i}', 0.)
    return p


def factory(version, z, r, phi, ms=MS, ns=NS, **kw):
    f = getattr(ff, f'brzphi_3d_producer_giant_function_v{version}')
    return f(z, r, phi, 0, 0, 0, 0, 0, 0, L, ms, ns, 0, 0, 0, Z0, **kw)


def test_v1010_matches_v1008_without_k0():
    z, r, phi, x, y = points()
    rng = np.random.default_rng(1)
    p = base_params(rng, ks={'k1': 1., 'k2': -2., 'k3': 3., 'k4': 0.5, 'k5': 0.1, 'k6': -0.2, 'k7': 0.3})
    b8 = factory(1008, z, r, phi)(z, r, phi, x, y, **p)
    b10 = factory(1010, z, r, phi)(z, r, phi, x, y, **p)
    np.testing.assert_allclose(b10, b8, rtol=1e-12, atol=1e-12)


def test_v1010_n_list_subset_matches_zeroed_orders():
    z, r, phi, x, y = points()
    rng = np.random.default_rng(2)
    p = base_params(rng)
    keep = [0, 2, 3]
    p_sub = {k: v for k, v in p.items() if not (k[1:4] == 'c1_' and int(k.split('_')[-1]) not in keep)}
    p_zero = {k: (0. if (k[1:4] == 'c1_' and int(k.split('_')[-1]) not in keep) else v) for k, v in p.items()}
    b_sub = factory(1010, z, r, phi, n_list_c1=keep)(z, r, phi, x, y, **p_sub)
    b_full = factory(1010, z, r, phi)(z, r, phi, x, y, **p_zero)
    np.testing.assert_allclose(b_sub, b_full, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('n', [1, 2, 3, 6, 10])
def test_k0_is_2d_multipole(n):
    # normal: By + i Bx = B_n ((x+iy)/R)^(n-1); skew: By + i Bx = i A_n ((x+iy)/R)^(n-1)
    z, r, phi, x, y = points()
    Bn, An = 1.7, -0.4
    w = ((x + 1j*y)/RREF)**(n-1)
    c = Bn*w + 1j*An*w
    By, Bx = c.real, c.imag
    Br = Bx*np.cos(phi) + By*np.sin(phi)
    Bphi = -Bx*np.sin(phi) + By*np.cos(phi)
    p = {'z0': Z0, f'Ek0_{n}': An, f'Fk0_{n}': Bn}
    p.update({f'k{i}': 0. for i in range(1, 11)})
    f = factory(1010, z, r, phi, ms=0, ns=0, n_list_k0=[n], R_ref=RREF)
    out = f(z, r, phi, x, y, **p)
    N = len(z)
    np.testing.assert_allclose(out[:N], Br, atol=1e-12)
    np.testing.assert_allclose(out[N:2*N], 0., atol=1e-15)
    np.testing.assert_allclose(out[2*N:], Bphi, atol=1e-12)


def test_design_matrix_matches_unit_steps():
    z, r, phi, x, y = points(N=200)
    rng = np.random.default_rng(3)
    nk0 = [1, 2, 5]
    f = factory(1010, z, r, phi, n_list_c1=[0, 1, 3], n_list_k0=nk0)
    p = base_params(rng, n_list=[0, 1, 3], n_list_k0=nk0)
    names = [k for k in p if k != 'z0' and not (k[0] in 'CD' and k.endswith('_0'))]
    J = f.design_matrix(names, z, r, phi, x, y)
    p0 = {k: (0. if k in names else v) for k, v in p.items()}
    off = f(z, r, phi, x, y, **p0)
    for j, k in enumerate(names):
        p0[k] = 1.
        col = f(z, r, phi, x, y, **p0) - off
        p0[k] = 0.
        np.testing.assert_allclose(J[:, j], col, rtol=1e-10, atol=1e-12, err_msg=k)


def make_df(n_list_k0, truth, N=1500, seed=11, noise=0.01):
    z, r, phi, x, y = points(N=N, seed=seed, n_axis=30)
    p = {'z0': Z0}
    p.update({f'k{i}': 0. for i in range(1, 11)})
    p.update(truth)
    f = factory(1010, z, r, phi, ms=0, ns=0, n_list_k0=n_list_k0, R_ref=RREF)
    out = f(z, r, phi, x, y, **p)
    rng = np.random.default_rng(seed+1)
    out = out + rng.normal(0, noise, len(out))
    N_ = len(z)
    return pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'R': r, 'Phi': phi,
                         'Br': out[:N_], 'Bz': out[N_:2*N_], 'Bphi': out[2*N_:]})


@pytest.mark.parametrize('method', ['linear', 'leastsq'])
def test_fit_recovers_k0_harmonics(method, tmp_path):
    nk0 = [1, 2, 6, 10]
    truth = {'Ek0_1': 0.3, 'Fk0_1': 2.0, 'Ek0_2': 0.1, 'Fk0_2': 6.6, 'Ek0_6': 0., 'Fk0_6': 0.02,
             'Ek0_10': 0., 'Fk0_10': -0.01}
    df = make_df(nk0, truth)
    cp = cfg_params(pitch1=0, ms_h1=0, ns_h1=0, pitch2=0, ms_h2=0, ns_h2=0,
                    length1=L, ms_c1=0, ns_c1=0, length2=0, ms_c2=0, ns_c2=0,
                    ks_dict={'k1': [0., False], 'k2': [0., False], 'k3': [0., True], 'k4': [0., False]},
                    bs_tuples=None, bs_bounds=None, loss='linear', version=1010, method=method,
                    noise=0.01, z0=Z0, n_list_k0=nk0, R_ref=RREF)
    ffit = FieldFitter(df)
    ffit.pickle_path = str(tmp_path) + '/'
    ffit.fit(cp, cfg_pickle(False, False, 'x', 'x', False))
    for k, v in truth.items():
        assert abs(ffit.params[k].value - v) < 5*ffit.params[k].stderr + 1e-6, k
    assert 0.8 < ffit.result.redchi < 1.2


def test_degeneracy_guard(tmp_path):
    df = make_df([1], {'Ek0_1': 0.3, 'Fk0_1': 2.0})
    cp = cfg_params(pitch1=0, ms_h1=0, ns_h1=0, pitch2=0, ms_h2=0, ns_h2=0,
                    length1=L, ms_c1=0, ns_c1=0, length2=0, ms_c2=0, ns_c2=0,
                    ks_dict={'k1': [0., True]}, bs_tuples=None, bs_bounds=None, loss='linear',
                    version=1010, method='linear', noise=0.01, z0=Z0, n_list_k0=[1])
    ffit = FieldFitter(df)
    ffit.pickle_path = str(tmp_path) + '/'
    with pytest.raises(ValueError, match='degenerate'):
        ffit.fit(cp, cfg_pickle(False, False, 'x', 'x', False))


@pytest.mark.parametrize('n_list', [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 10, 14], [2, 6, 10]])
@pytest.mark.parametrize('n_threads', [1, 8])
def test_v1010_bessel_setup_bit_identical(n_list, n_threads):
    from scipy import special
    z, r, phi, x, y = points(N=500)
    cms = 2*np.pi*(np.arange(7)+1)/L
    iv = np.zeros((len(cms), len(n_list), len(r)))
    ivp = np.zeros_like(iv)
    ff._v1010_iv_ivp(cms, r, n_list, iv, ivp, n_threads=n_threads)
    for m in range(len(cms)):
        for j, n in enumerate(n_list):
            assert np.array_equal(iv[m][j], special.iv(n, cms[m]*r))
            assert np.array_equal(ivp[m][j], special.ivp(n, cms[m]*r))


# ---- linear_lsq / method='linear_fast' ----
from mu2e.tools.linear_lsq import solve_linear, field_uncertainty


def test_solve_linear_matches_normal_equations():
    rng = np.random.default_rng(21)
    M, P = 300, 12
    J = rng.normal(size=(M, P)) * np.logspace(-3, 3, P)[np.newaxis, :]
    x_true = rng.normal(size=P)
    w = 1. / rng.uniform(0.5, 2.0, M)
    data = J @ x_true + rng.normal(size=M) / w
    sol = solve_linear(J, data, weights=w, verbose=False)
    A = J * w[:, None]
    C = np.linalg.inv(A.T @ A)
    x = C @ (A.T @ (data * w))
    np.testing.assert_allclose(sol['x'], x, rtol=1e-8)
    np.testing.assert_allclose(sol['covar'], C, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(sol['stderr'], np.sqrt(np.diag(C)), rtol=1e-8)
    assert sol['rank'] == P and not sol['null_flag'].any()
    Je = rng.normal(size=(50, P))
    np.testing.assert_allclose(field_uncertainty(Je, sol['covar'], chunk=7),
                               np.sqrt(np.einsum('ij,jk,ik->i', Je, C, Je)), rtol=1e-8)


def test_solve_linear_rank_deficient_flags_null_params():
    rng = np.random.default_rng(22)
    J = rng.normal(size=(200, 5))
    J[:, 4] = 2. * J[:, 3]  # exact degeneracy
    sol = solve_linear(J, rng.normal(size=200), verbose=False)
    assert sol['rank'] == 4
    assert sol['null_flag'][3] and sol['null_flag'][4] and not sol['null_flag'][:3].any()


def fit_k0_df(method, tmp_path, save=False):
    nk0 = [1, 2, 6, 10]
    truth = {'Ek0_1': 0.3, 'Fk0_1': 2.0, 'Ek0_2': 0.1, 'Fk0_2': 6.6, 'Ek0_6': 0., 'Fk0_6': 0.02,
             'Ek0_10': 0., 'Fk0_10': -0.01}
    df = make_df(nk0, truth)
    # add a small Bessel part too (ms = 3, n = 0..2) so the design matrix mixes term types
    cp = cfg_params(pitch1=0, ms_h1=0, ns_h1=0, pitch2=0, ms_h2=0, ns_h2=0,
                    length1=L, ms_c1=3, ns_c1=3, length2=0, ms_c2=0, ns_c2=0,
                    ks_dict={'k1': [0., False], 'k2': [0., False], 'k3': [0., True], 'k4': [0., False],
                             'k5': [0., True]},
                    bs_tuples=None, bs_bounds=None, loss='linear', version=1010, method=method,
                    noise=0.01, z0=Z0, n_list_k0=nk0, R_ref=RREF)
    ffit = FieldFitter(df)
    ffit.pickle_path = str(tmp_path) + '/'
    ffit.fit(cp, cfg_pickle(False, save, 'x', 'x', False))
    return ffit


def test_linear_fast_matches_lmfit(tmp_path):
    f_lm = fit_k0_df('linear', tmp_path)       # lmfit leastsq polish -> lmfit covar / stderr / correl
    f_fa = fit_k0_df('linear_fast', tmp_path)
    vn = f_lm.result.var_names
    assert vn == f_fa.result.var_names
    for k in vn:
        np.testing.assert_allclose(f_fa.params[k].value, f_lm.params[k].value, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(f_fa.params[k].stderr, f_lm.params[k].stderr, rtol=1e-3)
    np.testing.assert_allclose(f_fa.result.redchi, f_lm.result.redchi, rtol=1e-8)
    np.testing.assert_allclose(f_fa.result.covar, f_lm.result.covar, rtol=2e-3,
                               atol=1e-3*np.abs(f_lm.result.covar).max())
    for i, a in enumerate(vn):
        for j, b in enumerate(vn):
            if i != j:
                assert abs(f_fa.result.correl_matrix[i, j] - f_lm.params[a].correl[b]) < 1e-3
    np.testing.assert_allclose(f_fa.result.eval_uncertainty(f_fa.params),
                               f_lm.result.eval_uncertainty(f_lm.params), rtol=2e-3, atol=1e-12)
    assert f_fa.result.linear_fast and f_fa.result.nvarys == len(vn) and f_fa.result.errorbars


def test_linear_fast_pickles(tmp_path):
    import pickle
    f = fit_k0_df('linear_fast', tmp_path, save=True)
    params = pickle.load(open(str(tmp_path) + '/x_results.p', 'rb'))
    cd = pickle.load(open(str(tmp_path) + '/x_results_correl.p', 'rb'))
    assert cd['variables'] == f.result.var_names
    assert cd['correl'].shape == (len(f.result.var_names),)*2
    for k in f.result.var_names:
        assert params[k].vary and params[k].stderr == f.params[k].stderr
