from numba import njit, float64, boolean
import numpy as np


@njit(float64(float64[::1], float64[::1]), cache=True)
def neg_mean_squared_error(y_true, y_pred):
    """https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html"""
    return - np.mean((y_pred - y_true) ** 2)


@njit(float64(float64[::1], float64[::1]), cache=True)
def r2_score(y_true, y_pred):
    """https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score"""
    y_mean = np.mean(y_true)
    total_sum_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_sum_squares / total_sum_squares)


@njit(float64(float64[::1], float64[::1], boolean), cache=True)
def calculate_score(y_true, y_pred, r2):
    if r2:
        return r2_score(y_true, y_pred)
    else:
        return neg_mean_squared_error(y_true, y_pred)
