from test_utils import evaluate, create_data, choose_cv_and_scoring
from numbaml.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.linear_model import LinearRegression as SKLLinearRegression
from sklearn.linear_model import RidgeCV as SKLRidgeCV
from sklearn.linear_model import Ridge as SKLRidge

import numpy as np
import time


def test_linear_regression(trials: int = 1):
    """this test aims to show the LinearRegression is equivalent to sci-kit learn LinearRegression"""
    for n in range(trials):
        print(f'---------------- CHECK LinearRegression - TRIAL {n + 1} ----------------')
        x, y, alphas = create_data(n_alphas=0)

        skl_model = SKLLinearRegression()
        skl_model.fit(x, y)
        skl_predict = skl_model.predict(x)

        model = LinearRegression()
        model.fit(x, y)
        predict = model.predict(x)

        print('predict:', evaluate(skl_predict, predict))
        print('coefs:', evaluate(skl_model.coef_, model.coef_))
        print('intercept:', evaluate(np.float64(skl_model.intercept_), np.float64(model.intercept_)))


def test_fit_intercept_false(trials: int = 1):
    """this test aims to show the fit_intercept=False parameter works"""
    for n in range(trials):
        print(f'---------------- CHECK fit_intercept=False - TRIAL {n + 1} ----------------')
        x, y, alphas = create_data(n_alphas=0)

        skl_model = SKLLinearRegression(fit_intercept=False)
        skl_model.fit(x, y)
        skl_predict = skl_model.predict(x)

        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        predict = model.predict(x)

        print('predict:', evaluate(skl_predict, predict))
        print('coefs:', evaluate(skl_model.coef_, model.coef_))
        print('intercept:', evaluate(np.float64(skl_model.intercept_), np.float64(model.intercept_)))


def test_ridge(trials: int = 1):
    """this test aims to show the Ridge is equivalent to sci-kit learn Ridge"""
    for n in range(trials):
        print(f'---------------- CHECK Ridge - TRIAL {n + 1} ----------------')
        x, y, alphas = create_data(n_alphas=1)

        skl_model = SKLRidge(alpha=alphas[0])
        skl_model.fit(x, y)
        skl_predict = skl_model.predict(x)

        model = Ridge(alpha=alphas[0])
        model.fit(x, y)
        predict = model.predict(x)

        print('predict:', evaluate(skl_predict, predict))
        print('coefs:', evaluate(skl_model.coef_, model.coef_))
        print('intercept:', evaluate(np.float64(skl_model.intercept_), np.float64(model.intercept_)))


def test_ridgecv(trials: int = 1):
    """this test to aims to show that RidgeCV is equivalent to sci-kit learn RidgeCV"""
    for n in range(trials):
        print(f'---------------- CHECK RidgeCV - TRIAL {n + 1} ----------------')
        x, y, alphas = create_data(n_alphas=5)
        cv, scoring = choose_cv_and_scoring(x.shape[0])

        start = time.time()
        skl_model = SKLRidgeCV(alphas=alphas, cv=cv, scoring=scoring)
        skl_model.fit(x, y)
        total_time = np.round(time.time() - start, 6)
        print(f'sklearn time: {total_time}')

        start = time.time()
        model = RidgeCV(alphas=alphas, cv=cv, scoring=scoring)
        model.fit(x, y)
        total_time = np.round(time.time() - start, 6)
        print(f'numba time: {total_time}')

        print('best alpha:', evaluate(np.float64(skl_model.alpha_), np.float64(model.alpha_)))
        print('best score:', evaluate(np.float64(skl_model.best_score_), np.float64(model.best_score_)))


def test_aloocv(trials: int = 1):
    """this test to aims to show that approx leave one out is equivalent to sci-kit learn"""
    for n in range(trials):
        print(f'---------------- CHECK ALOOCV - TRIAL {n + 1} ----------------')
        x, y, alphas = create_data(n_alphas=5)

        start = time.time()
        skl_model = SKLRidgeCV(alphas=alphas, cv=None)
        skl_model.fit(x, y)
        total_time = np.round(time.time() - start, 6)
        print(f'sklearn time: {total_time}')

        start = time.time()
        model = RidgeCV(alphas=alphas, cv=None)
        model.fit(x, y)
        total_time = np.round(time.time() - start, 6)
        print(f'numba time: {total_time}')

        print('best alpha:', evaluate(np.float64(skl_model.alpha_), np.float64(model.alpha_)))
        print('best score:', evaluate(np.float64(skl_model.best_score_), np.float64(model.best_score_)))
