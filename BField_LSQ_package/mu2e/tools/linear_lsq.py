'''Weighted linear least squares with full uncertainty information, for fit functions that are
linear in all free parameters (e.g. v1010 via its design_matrix).

For a model B = J p with data weights w (1/sigma), A = diag(w) J. Columns are normalised by
D = diag(||A_j||) to make the SVD well scaled: A D^-1 = U S V^T. Then

    p     = D^-1 V S^-1 U^T b                      (b = w * data)
    C     = D^-1 V S^-2 V^T D^-1 = (J^T W^2 J)^-1  (lmfit's covar with scale_covar=False)
    sigma = sqrt(diag(C)),  correl = C / (sigma sigma^T)

Singular values below rcond * S_max are truncated (pseudo-inverse); parameters with weight in the
truncated (null) directions are flagged, since their uncertainties are then only lower bounds.
The field uncertainty at any set of points with design matrix J_e is sqrt(diag(J_e C J_e^T)).
'''
import numpy as np
from scipy import linalg


def solve_linear(J, data, weights=None, offset=None, rcond=None, scale_covar=False,
                 compute_covar=True, overwrite_J=False, verbose=True):
    '''Solve min || w (J p + offset - data) ||^2 by SVD.

    Args:
        J: (M, P) design matrix (unweighted). Modified in place if overwrite_J.
        data: (M,) data vector.
        weights: (M,) weights (1/sigma); default ones.
        offset: (M,) fixed-parameter model contribution; default 0.
        rcond: relative singular value cut; default eps * max(M, P) (as numpy/scipy lstsq).
        scale_covar: multiply C by chi2_red (lmfit default); FieldFitter uses False.
        compute_covar: also return covar / stderr / correl.

    Returns:
        dict with x, chi2, redchi, ndata, nvarys, nfree, rank, cond, sv, resid_w (weighted
        residual w(model - data)), model (J x + offset) and, if compute_covar, covar, stderr,
        correl, null_flag (bool per parameter: touches a truncated direction).
    '''
    M, P = J.shape
    w = np.ones(M) if weights is None else np.asarray(weights, dtype=np.float64)
    off = np.zeros(M) if offset is None else np.asarray(offset, dtype=np.float64)
    b = (np.asarray(data, dtype=np.float64) - off) * w
    A = J if overwrite_J else J.copy()
    A *= w[:, np.newaxis]
    col = np.linalg.norm(A, axis=0)
    col[col == 0] = 1.
    A /= col[np.newaxis, :]
    # gesdd may overwrite A in place (Fortran-ordered input); keep a copy for the gesvd fallback then
    A_bak = A.copy() if A.flags['F_CONTIGUOUS'] else None
    try:
        U, S, Vt = linalg.svd(A, full_matrices=False, lapack_driver='gesdd', overwrite_a=True,
                              check_finite=False)
        driver = 'gesdd'
    except linalg.LinAlgError as e:
        print(f'solve_linear: WARNING gesdd failed ({e}); falling back to gesvd.', flush=True)
        if A_bak is not None:
            A = A_bak
        U, S, Vt = linalg.svd(A, full_matrices=False, lapack_driver='gesvd', overwrite_a=True,
                              check_finite=False)
        driver = 'gesvd'
    del A, A_bak
    if rcond is None:
        rcond = np.finfo(np.float64).eps * max(M, P)
    keep = S > rcond * S[0]
    rank = int(keep.sum())
    Uk = U[:, keep]
    Sk = S[keep]
    Vk = Vt[keep]
    Utb = Uk.T @ b
    x = (Vk.T @ (Utb / Sk)) / col
    fit_w = Uk @ Utb                 # weighted model at the optimum (projection of b)
    del U, Uk
    resid_w = fit_w - b
    chi2 = float(resid_w @ resid_w)
    nfree = M - rank
    out = {'x': x, 'chi2': chi2, 'ndata': M, 'nvarys': P, 'nfree': nfree,
           'redchi': chi2 / nfree if nfree > 0 else np.nan, 'rank': rank,
           'cond': float(S[0] / S[keep][-1]), 'sv': S, 'resid_w': resid_w, 'svd_driver': driver,
           'model': fit_w / w + off}
    if compute_covar:
        Vs = Vk.T / Sk[np.newaxis, :]    # (P, rank)
        C = (Vs @ Vs.T) / np.outer(col, col)
        if scale_covar:
            C *= out['redchi']
        stderr = np.sqrt(np.clip(np.diag(C), 0., None))
        with np.errstate(divide='ignore', invalid='ignore'):
            correl = C / np.outer(stderr, stderr)
        correl[~np.isfinite(correl)] = 0.
        if rank < P:
            Vnull = Vt[~keep]
            null_flag = np.sum(Vnull**2, axis=0) > 1e-6
        else:
            null_flag = np.zeros(P, dtype=bool)
        out.update({'covar': C, 'stderr': stderr, 'correl': correl, 'null_flag': null_flag})
    if verbose:
        print(f'solve_linear: M={M}, P={P}, rank={rank}, cond(column-normalised)={out["cond"]:0.3e}, '
              +f'chi2={chi2:0.6e}, redchi={out["redchi"]:0.6e}, svd={driver}'
              +(f', {int(out["null_flag"].sum())} params touch null directions' if compute_covar and rank < P else ''))
    return out


def field_uncertainty(J_eval, covar, chunk=20000):
    '''sqrt(diag(J_eval C J_eval^T)) for a (M, P) design matrix at the evaluation points,
    computed in row chunks to bound memory.'''
    M = J_eval.shape[0]
    out = np.empty(M)
    for i in range(0, M, chunk):
        Jc = J_eval[i:i+chunk]
        out[i:i+chunk] = np.sqrt(np.clip(np.einsum('ij,ij->i', Jc @ covar, Jc), 0., None))
    return out


def t_scale(ndata, nvarys, sigma=1):
    '''Student-t scale lmfit's eval_uncertainty applies for a given sigma level.'''
    from scipy.special import erf
    from scipy.stats import t
    prob = erf(sigma / np.sqrt(2))
    return t.ppf((prob + 1) / 2., ndata - nvarys)
