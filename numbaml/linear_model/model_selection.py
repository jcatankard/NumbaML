from .fit import fit, create_penalty_matrix
from .metrics import r2_score, neg_mean_squared_error
from .predict import predict

from numba import njit, float64, int64, types, prange, boolean
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


@njit(float64(float64[::1], float64[::1], boolean), cache=True)
def calculate_score(y_true, y_pred, r2):
    return r2_score(y_true, y_pred) if r2 else neg_mean_squared_error(y_true, y_pred)


@njit(types.UniTuple(float64, 2)(float64[:, ::1], float64[:, ::1], float64[::1], boolean, int64, boolean),
      parallel=True, cache=True)
def find_alpha_kfolds(x, y, alphas, fit_intercept, cv, r2):
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

            weights = fit(train_x, train_y, a, fit_intercept)
            preds = predict(test_x, weights)

            scores[j] = calculate_score(test_y.flatten(), preds.flatten(), r2)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = a

    return best_alpha, best_score


@njit(float64[::1](float64[:, ::1], float64[:, ::1], float64, boolean), cache=True)
def approximate_leave_one_out_errors(x, y, l2_penalty, fit_intercept):
    """https://medium.com/@jcatankard_76170/efficient-leave-one-out-cross-validation-f1dee3b68dfe"""

    weights = fit(x, y, l2_penalty, fit_intercept)
    preds = predict(x, weights)
    residuals = y - preds

    penalty = create_penalty_matrix(l2_penalty, x.shape[1], fit_intercept)

    h = np.diag(x @ np.linalg.inv(x.T @ x + penalty) @ x.T)
    return (residuals / (1 - h).reshape(-1, 1)).flatten()


@njit(types.UniTuple(float64, 2)(float64[:, ::1], float64[:, ::1], float64[::1], boolean), parallel=True, cache=True)
def find_alpha_loo(x, y, alphas, fit_intercept):

    all_scores = np.zeros(alphas.size, dtype=np.float64)
    for i in prange(alphas.size):
        errors = approximate_leave_one_out_errors(x, y, alphas[i], fit_intercept)
        # calculate neg mean squared error
        all_scores[i] = - np.mean(errors ** 2)

    best_alpha = alphas[np.argmax(all_scores)]
    best_score = np.max(all_scores)
    return best_alpha, best_score
