from numbaml.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np


if __name__ == '__main__':
    X, y = make_regression(n_samples=150, n_features=10, noise=10)
    significance_level = .1

    m = LinearRegression(fit_intercept=True)
    m.fit(X, y)

    ci = m.conf_int(sig=significance_level, bootstrap_method=True, bootstrap_iterations=100000)
    params = np.vstack((ci[:, 0], m.params_, ci[:, 1])).T
    gap1 = ci[:, 1] - ci[:, 0]
    print(params.round(2))


    ci = m.conf_int(sig=significance_level)
    params = np.vstack((ci[:, 0], m.params_, ci[:, 1])).T
    gap2 = ci[:, 1] - ci[:, 0]
    print(params.round(2))

    print(np.stack([gap1, gap2]).T.round(2))


