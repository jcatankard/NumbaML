from numba import njit, int64, float64, boolean
import numpy as np


@njit(float64[:, ::1](float64, int64, boolean), cache=True)
def create_penalty_matrix(l2_penalty, n_features, fit_intercept):
    """n_features includes intercept"""
    identity = np.identity(n_features)
    # set top corner to zero as we don't penalize intercept
    identity[0][0] = 0 if fit_intercept else 1
    return l2_penalty * identity


@njit(float64[::1](float64[:, ::1], float64[::1], float64, boolean), cache=True)
def fit(x, y, l2_penalty, fit_intercept):
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
    :param fit_intercept: if l2 penalty is 0, this has no impact
    :return: coefficients, intercept
    """
    penalty = create_penalty_matrix(l2_penalty, x.shape[1], fit_intercept)
    weights = np.linalg.inv(x.T @ x + penalty) @ x.T @ y
    return weights
