from numbaml.metrics import residual_mean_squared_error
from numba import njit, int64, float64
from numbaml.predict import predict
from scipy import stats
import numpy as np


@njit(int64(float64[:, ::1]), cache=True)
def residual_degrees_of_freedom(x):
    """
    The residual degree of freedom defined as the number of observations minus
    the rank of the regressor matrix (linearly independent features)
    """
    n_samples = x.shape[0]
    rank = np.linalg.matrix_rank(x)
    return np.max(np.array([n_samples - rank, 1]))


@njit(float64[::1](float64[:, ::1], float64), cache=True)
def parameter_standard_errors(x, scale):
    """
    The variance/covariance matrix can be of a linear contrast of the estimated parameters or all params multiplied by
    scale which will usually be an estimate of sigma².
    Scale will typically be the mean square error from the estimated model (sigma²).
    - off-diagonal elements, Cij, represent the covariance between the ith and jth estimated regression coefficients
    - symmetrical elements Cjj represent variance of the jth regression coefficient
    """
    c = np.linalg.inv(x.T @ x) * scale
    v = np.diag(c)
    return np.sqrt(v)


@njit(float64[:, ::1](float64[:, ::1], float64[::1], float64[::1], int64, float64), cache=True)
def parameter_confidence_intervals(x, y, weights, dof, t_value):
    """
    https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.conf_int.html
    https://www.statsmodels.org/stable/_modules/statsmodels/regression/linear_model.html
    https://github.com/statsmodels/statsmodels/blob/main/statsmodels/base/model.py
    lower, upper = params - q * bse, params + q * bse
    """
    y_fitted = predict(x, weights)
    mse = residual_mean_squared_error(y, y_fitted, dof)
    se = parameter_standard_errors(x, mse)
    gap = t_value * se
    return np.stack((weights - gap, weights + gap))


def calculate_t_value(alpha, dof):
    return stats.t.ppf(1 - alpha / 2, dof)


def conf_int(x, y, params, alpha):
    dof = residual_degrees_of_freedom(x)
    t_value = calculate_t_value(alpha=alpha, dof=dof)
    return parameter_confidence_intervals(x, y, params, dof, t_value).T
