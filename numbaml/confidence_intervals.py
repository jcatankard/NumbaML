from numbaml.metrics import residual_mean_squared_error
from numba import njit, int64, float64, prange
from numbaml.predict import predict
from numbaml.fit import fit
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


@njit(float64[:, ::1](float64[:, ::1], float64[:, ::1], float64[:, ::1], int64, float64), cache=True)
def parameter_confidence_intervals(x, y, weights, dof, t_value):
    """
    https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.conf_int.html
    https://www.statsmodels.org/stable/_modules/statsmodels/regression/linear_model.html
    https://github.com/statsmodels/statsmodels/blob/main/statsmodels/base/model.py
    lower, upper = params - q * bse, params + q * bse
    """
    y_fitted = predict(x, weights)
    mse = residual_mean_squared_error(y.flatten(), y_fitted.flatten(), dof)
    se = parameter_standard_errors(x, mse)
    gap = t_value * se
    return np.stack((weights.flatten() - gap, weights.flatten() + gap))


def calculate_t_value(sig, dof):
    return stats.t.ppf(1 - sig / 2, dof)


def conf_int_parameter_method(x, y, params, sig):
    dof = residual_degrees_of_freedom(x)
    t_value = calculate_t_value(sig=sig, dof=dof)
    return parameter_confidence_intervals(x, y, params, dof, t_value).T


@njit(float64[:, ::1](float64[:, ::1], float64), cache=True, parallel=True)
def calculate_ci(samples, sig):
    sig_levels = np.array([sig / 2, 1 - sig / 2]) * 100
    samples = samples.T
    n_features = samples.shape[0]
    percentiles = np.zeros(shape=(n_features, 2), dtype=np.float64)
    for i in prange(n_features):
        percentiles[i] = np.percentile(samples[i], sig_levels)
    return percentiles


@njit(float64[:, ::1](float64[:, ::1], float64[:, ::1], int64), cache=True, parallel=True)
def create_param_resamples(x, y, n_iterations):
    n_samples, n_features = x.shape
    index_ = np.arange(n_samples)
    samples = np.zeros(shape=(n_iterations, n_features), dtype=np.float64)
    for i in prange(n_iterations):
        random_index = np.random.choice(index_, n_samples, replace=True)
        # l2 penalty always 0 as conf intervals for OLS only - fit intercept has no affect if l2 penalty=0
        samples[i] = fit(x[random_index], y[random_index], np.float64(0), True)
    return samples


@njit(float64[:, ::1](float64[:, ::1], float64[:, ::1], float64, int64), cache=True)
def conf_int_bootstrap_method(x, y, sig, n_iterations):
    samples = create_param_resamples(x, y, n_iterations)
    upper_lower_bounds = calculate_ci(samples, sig)
    return upper_lower_bounds
