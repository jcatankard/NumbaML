from numba import njit, float64
import numpy as np


@njit(float64[::1](float64[:, ::1], float64[::1], float64), cache=True)
def predict(x, weights, bias):
    return np.dot(x, weights) + bias
