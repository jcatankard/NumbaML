from numba import njit, int64, float64, types
import numpy as np


@njit(float64[:, ::1](float64[:, ::1]), cache=True)
def add_intercept(x):
    n_samples, n_features = x.shape
    intercept = np.ones((n_samples, 1), dtype=np.float64)
    x_w_intercept = np.zeros(shape=(n_samples, 1 + n_features), dtype=np.float64)
    x_w_intercept[:, : 1] = intercept
    x_w_intercept[:, 1:] = x
    return x_w_intercept


@njit(float64[:, ::1](float64, int64), cache=True)
def create_penalty_matrix(l2_penalty, n_features):
    """n_features includes intercept"""
    identity = np.identity(n_features)
    # set top corner to zero as we don't penalize intercept
    identity[0][0] = 0
    return l2_penalty * identity


@njit(types.Tuple((float64[::1], float64))(float64[:, ::1], float64[::1], float64), cache=True)
def fit(x, y, l2_penalty):
    """
    Solution 1 (solve regression formula for B)
    solve for B: y = X.B -> Xt.y = (Xt.X).B -> Xt.(Xt.X)^-1.y = (Xt.X).(Xt.X)^-1.B = I.B = B
    add penalty: Xt.(Xt.X + A)^-1.y (we don't penalize intercept, see code)

    Solution 2 (minimise SSE) web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
    error = ypi - yi = b0 + bi.xi + e - yi
    SSS = E(b0 + bi.xi - yi)^2
    SSE = (matrix: intercept->B) (B.X - y)^2 = (B.X - y).(B.X - y)t = B.X.Bt.Xt - y.Bt.Xt - yt.B.X + y.yt
    = yt.y + B.X.Bt.Xt - 2.y.Bt.Xt (y.Bt.Xt == yt.B.X)
    dSSE/db = 2.B.X.Xt - 2.y.Xt = -2.Xt.(y - B.X)
    minimise SSE: 0 = -2.Xt.(y - B.X) = -Xt.y + B.X.Xt -> Xt.y = B.X.Xt -> (X.Xt)^-1.Xt.y = B.(X.Xt).(X.Xt)^-1 = B

    :param x: independent variables
    :param y: dependent target
    :param l2_penalty: regularization penalty
    :return: coefficients, intercept
    """
    x = add_intercept(x)
    penalty = create_penalty_matrix(l2_penalty, n_features=x.shape[1])
    weights = np.linalg.inv(x.T @ x + penalty) @ x.T @ y
    return weights[1:], weights[0]
