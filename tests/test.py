from numbaml.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.linear_model import LinearRegression as SKLLinearRegression
from sklearn.linear_model import RidgeCV as SKLRidgeCV
from sklearn.linear_model import Ridge as SKLRidge
from sklearn.datasets import make_regression

import numpy as np
import time


def test(a1, a2, precision: int = 6) -> bool:
    return np.array_equal(a1.round(precision), a2.round(precision))


def create_data(n_alphas: int):
    n_samples = np.random.randint(10, 1000)
    max_features = 100 if n_samples // 2 > 100 else n_samples // 2
    n_features = np.random.randint(1, max_features)
    noise = np.random.randint(10, 100)
    x, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)

    alphas = np.array([.001, .01, .1, .5, .9, .99, 1., 1.1, 2.5, 5., 10.])
    alphas = np.random.choice(alphas, n_alphas, replace=False)

    q = f"""
    n_samples = {n_samples}
    n_features = {n_features}
    noise = {noise}
    alphas = {alphas}
    """
    print(q)
    return x, y, alphas


def choose_cv_and_scoring(n_samples: int):
    cv_max = 10 if n_samples > 10 else n_samples
    cv = np.random.randint(2, cv_max + 1)
    scoring = np.random.choice(['r2', 'neg_mean_squared_error'])
    q = f"""
    cv = {cv}
    scoring = {scoring}
    """
    print(q)
    return cv, scoring


def check_linear_regression(trials: int = 1):
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

        print('predict:', test(skl_predict, predict))
        print('coefs:', test(skl_model.coef_, model.coef_))
        print('intercept:', test(np.float64(skl_model.intercept_), np.float64(model.intercept_)))


def check_ridge(trials: int = 1):
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

        print('predict:', test(skl_predict, predict))
        print('coefs:', test(skl_model.coef_, model.coef_))
        print('intercept:', test(np.float64(skl_model.intercept_), np.float64(model.intercept_)))

        print(model.coef_ / skl_model.coef_)


def check_ridgecv(trials: int = 1):
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

        print('best alpha:', test(np.float64(skl_model.alpha_), np.float64(model.alpha_)))
        print('best score:', test(np.float64(skl_model.best_score_), np.float64(model.best_score_)))


def check_aloocv(trials: int = 1):
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

        print('best alpha:', test(np.float64(skl_model.alpha_), np.float64(model.alpha_)))
        print('best score:', test(np.float64(skl_model.best_score_), np.float64(model.best_score_)))


check_ridge(5)
check_linear_regression(5)
check_ridgecv(5)
check_aloocv(5)
