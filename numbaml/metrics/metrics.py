from numba import njit, float64, boolean
import numpy as np


@njit(float64(float64[::1], float64[::1]), cache=True)
def sum_square_residuals(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


@njit(float64(float64[::1], float64[::1], boolean), cache=True)
def mean_squared_error(y_true, y_pred, squared: bool = True):
    """https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html"""
    mse = sum_square_residuals(y_true, y_pred) / y_true.size
    return mse if squared else mse ** .5


@njit(float64(float64[::1], float64[::1]), cache=True)
def neg_mean_squared_error(y_true, y_pred):
    return - mean_squared_error(y_true, y_pred, squared=True)


@njit(float64(float64[::1], float64[::1]), cache=True)
def r2_score(y_true, y_pred):
    """https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score"""
    y_mean = np.mean(y_true)
    total_sum_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_squares = sum_square_residuals(y_true, y_pred)
    return 1 - (residual_sum_squares / total_sum_squares)
