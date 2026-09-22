import numpy as np
import pandas as pd

# numerical derivatives
def calc_jacobian_numerical(df_):
    '''
    df_ should be a pd.DataFrame containing repetitions of the following pattern order:
    0: nominal, 1: +Z, 2: -Z, 3: +Y, 4: -Y, 5: +X, 6: -X
    returns: J, where row indicates B component, column indicates numerical component
    '''
    x, y, z, Bx, By, Bz = df_[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']].values.T
    J = np.zeros((len(x)//7, 3, 3))
    # dx
    J[:, 0, 0] = (Bx[5::7] - Bx[6::7]) / (x[5::7] - x[6::7]) # dBx
    J[:, 1, 0] = (By[5::7] - By[6::7]) / (x[5::7] - x[6::7]) # dBy
    J[:, 2, 0] = (Bz[5::7] - Bz[6::7]) / (x[5::7] - x[6::7]) # dBz
    # dy
    J[:, 0, 1] = (Bx[3::7] - Bx[4::7]) / (y[3::7] - y[4::7]) # dBx
    J[:, 1, 1] = (By[3::7] - By[4::7]) / (y[3::7] - y[4::7]) # dBy
    J[:, 2, 1] = (Bz[3::7] - Bz[4::7]) / (y[3::7] - y[4::7]) # dBz
    # dz
    J[:, 0, 2] = (Bx[1::7] - Bx[2::7]) / (z[1::7] - z[2::7]) # dBx
    J[:, 1, 2] = (By[1::7] - By[2::7]) / (z[1::7] - z[2::7]) # dBy
    J[:, 2, 2] = (Bz[1::7] - Bz[2::7]) / (z[1::7] - z[2::7]) # dBz

    # T / m, when coming directly from helicalc
    return J

def scale_J(J, T_to_Gauss=False, m_to_mm=False):
    if T_to_Gauss:
        J = J*1e4
    if m_to_mm:
        J = J*1e-3
    return J

def calc_div(J):
    #return np.sum(np.diag(J))
    return J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]

def calc_curl(J):
    curl_x = J[:, 2, 1] - J[:, 1, 2]
    curl_y = J[:, 0, 2] - J[:, 2, 0]
    curl_z = J[:, 1, 0] - J[:, 0, 1]
    return np.array([curl_x, curl_y, curl_z]).T

def div_and_curl_calculations(df, T_to_Gauss=False, m_to_mm=False):
    J = calc_jacobian_numerical(df)
    J = scale_J(J, T_to_Gauss, m_to_mm)
    div = calc_div(J)
    curl_vec = calc_curl(J)
    curl = np.linalg.norm(curl_vec, axis=1)
    # set in df
    df_nom = df.iloc[::7].copy()
    df_nom.reset_index(drop=True, inplace=True)
    df_nom.loc[:, 'divB'] = div
    df_nom.loc[:, 'curlB'] = curl
    df_nom.loc[:, 'curlB_x'] = curl_vec[:, 0]
    df_nom.loc[:, 'curlB_y'] = curl_vec[:, 1]
    df_nom.loc[:, 'curlB_z'] = curl_vec[:, 2]
    return df_nom, J


# ---------------------------------------------------------------------------
# 4th-order central differences
# ---------------------------------------------------------------------------
# calc_jacobian_numerical above is 2nd order (7 points per node).  Verifying
# that a field is Maxwellian is much sharper with two orders: the residuals must
# fall as h^2 and h^4 respectively until they hit the roundoff floor.  A residual
# that stops falling with h indicates an open winding or an interpolation
# artefact rather than a discretisation error.
#
# Point pattern, 13 per node:
#   0: nominal
#   1..4:   +Z, -Z, +2Z, -2Z
#   5..8:   +Y, -Y, +2Y, -2Y
#   9..12:  +X, -X, +2X, -2X

def add_points_for_J_4th(df, dxyz=0.001):
    '''Expand each row into the 13-point stencil used by calc_jacobian_4th.'''
    x0s, y0s, z0s = df[['X', 'Y', 'Z']].values.T
    d = dxyz
    xs = np.concatenate(np.array([x0s] * 9 +
                                 [x0s + d, x0s - d, x0s + 2 * d, x0s - 2 * d]).T)
    ys = np.concatenate(np.array([y0s] * 5 +
                                 [y0s + d, y0s - d, y0s + 2 * d, y0s - 2 * d] +
                                 [y0s] * 4).T)
    zs = np.concatenate(np.array([z0s] +
                                 [z0s + d, z0s - d, z0s + 2 * d, z0s - 2 * d] +
                                 [z0s] * 8).T)
    out = {'X': xs, 'Y': ys, 'Z': zs}
    if 'HP' in df.columns:
        out['HP'] = np.concatenate(np.array(13 * [df.HP.values]).T)
    return pd.DataFrame(out)


def calc_jacobian_4th(df_):
    '''4th-order Jacobian from the 13-point stencil of add_points_for_J_4th.

    d/dq f = (-f(+2h) + 8 f(+h) - 8 f(-h) + f(-2h)) / (12 h)
    '''
    x, y, z, Bx, By, Bz = df_[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']].values.T
    n = len(x) // 13
    J = np.zeros((n, 3, 3))
    # (axis index, stencil offsets for +h, -h, +2h, -2h, coordinate array)
    axes = [(0, 9, 10, 11, 12, x), (1, 5, 6, 7, 8, y), (2, 1, 2, 3, 4, z)]
    for ax, ip, im, i2p, i2m, q in axes:
        h = q[ip::13] - q[::13]
        for comp, B in enumerate((Bx, By, Bz)):
            J[:, comp, ax] = (-B[i2p::13] + 8 * B[ip::13]
                              - 8 * B[im::13] + B[i2m::13]) / (12 * h)
    return J


def div_and_curl_calculations_4th(df, T_to_Gauss=False, m_to_mm=False):
    '''div/curl from the 13-point 4th-order stencil; mirrors the 2nd-order call.'''
    J = calc_jacobian_4th(df)
    J = scale_J(J, T_to_Gauss, m_to_mm)
    div = calc_div(J)
    curl_vec = calc_curl(J)
    curl = np.linalg.norm(curl_vec, axis=1)
    df_nom = df.iloc[::13].copy()
    df_nom.reset_index(drop=True, inplace=True)
    df_nom.loc[:, 'divB'] = div
    df_nom.loc[:, 'curlB'] = curl
    df_nom.loc[:, 'curlB_x'] = curl_vec[:, 0]
    df_nom.loc[:, 'curlB_y'] = curl_vec[:, 1]
    df_nom.loc[:, 'curlB_z'] = curl_vec[:, 2]
    return df_nom, J
