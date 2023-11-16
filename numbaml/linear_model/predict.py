from numba import njit, float64


@njit(float64[:, ::1](float64[:, ::1], float64[:, ::1]), cache=True)
def predict(x, weights):
    return x @ weights
