from sklearn.datasets import make_regression
import numpy as np


def evaluate(a1, a2, precision: int = 6) -> bool:
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
