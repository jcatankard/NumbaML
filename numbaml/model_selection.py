from numbaml.fit import fit, add_intercept, create_penalty_matrix
from numba import njit, float64, int64, types, prange, boolean
from numbaml.metrics import calculate_score
from numbaml.predict import predict
import numpy as np


@njit(boolean[:, ::1](int64, int64), cache=True)
def kfold(n_samples, n_splits):
    """scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold"""
    first_splits = n_samples % n_splits
    test_flags = np.zeros(shape=(n_splits, n_samples), dtype=boolean)
    start = 0
    for i in range(n_splits):
        size = n_samples // n_splits + 1 if i < first_splits else n_samples // n_splits
        end = start + size
        test_flags[i][start: end] = np.ones(size, dtype=boolean)
        start += size
    return test_flags


@njit(types.UniTuple(float64, 2)(float64[:, ::1], float64[::1], float64[::1], int64, boolean),
      parallel=True, cache=True)
def find_alpha(x, y, alphas, cv, r2):
    n_samples, n_features = x.shape

    best_score = -np.inf
    best_alpha = np.float64(0)

    all_test_flags = kfold(n_samples, n_splits=cv)

    for i in range(alphas.size):
        a = alphas[i]
        scores = np.zeros(cv, dtype=np.float64)
        for j in prange(cv):

            is_test = all_test_flags[j]
            test_x, test_y = x[is_test], y[is_test]
            train_x, train_y = x[~is_test], y[~is_test]

            coefs, intercept = fit(train_x, train_y, l2_penalty=a)
            preds = predict(test_x, coefs, intercept)

            scores[j] = calculate_score(test_y, preds, r2)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = a

    return best_alpha, best_score


@njit(float64(float64[:, ::1], float64[::1], float64), cache=True)
def approximate_leave_one_out(x, y, l2_penalty):

    coefs, intercept = fit(x, y, l2_penalty=l2_penalty)
    preds = predict(x, coefs, intercept)
    residuals = y - preds

    x = add_intercept(x)
    penalty = create_penalty_matrix(l2_penalty, n_features=x.shape[1])

    h = np.diag(x @ np.linalg.inv(x.T @ x + penalty) @ x.T)
    mean_squared_errors = (residuals / (1 - h)) ** 2
    return - np.mean(mean_squared_errors)


@njit(types.UniTuple(float64, 2)(float64[:, ::1], float64[::1], float64[::1]), parallel=True, cache=True)
def find_alpha_aloocv(x, y, alphas):

    all_scores = np.zeros(alphas.size, dtype=np.float64)
    for i in prange(alphas.size):
        all_scores[i] = approximate_leave_one_out(x, y, l2_penalty=alphas[i])

    best_alpha = alphas[np.argmax(all_scores)]
    best_score = np.max(all_scores)
    return best_alpha, best_score
