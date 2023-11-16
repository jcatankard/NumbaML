from numbaml.linear_model._model_selection import find_alpha_loo, find_alpha_kfolds
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import time


if __name__ == '__main__':
    trials = 10
    results = np.zeros(shape=(trials, 9))
    alphas = np.array([.1, .5, 1., 5., 10.])

    for i in range(trials):
        n_samples = np.random.randint(100, 250)
        n_features = np.random.randint(10, 25)
        noise = np.random.randint(1, 10)
        X, y = make_regression(n_samples=n_samples,
                               n_features=n_features,
                               noise=noise,
                               n_targets=1
                               )
        y = y.reshape(y.shape[0], -1)
        fit_intercept = True
        start_cv = time.time()
        alpha_cv, score_cv = find_alpha_kfolds(X, y, alphas, cv=n_samples, r2=False, fit_intercept=fit_intercept)
        time_cv = time.time() - start_cv

        start_gcv = time.time()
        alpha_gcv, score_gcv = find_alpha_loo(X, y, alphas, fit_intercept=fit_intercept)
        time_gcv = time.time() - start_gcv

        results[i] = np.array([n_samples, n_features, noise,
                               alpha_cv, score_cv, time_cv,
                               alpha_gcv, score_gcv, time_gcv
                               ])
    results = pd.DataFrame(
        data=results,
        columns=['n_samples', 'n_features', 'noise',
                 'alpha_loo', 'score_loo', 'time_loo',
                 'alpha_aloo', 'score_aloo', 'time_aloo'
                 ]
    )
    results = results.astype({'n_samples': int, 'n_features': int})
    results['alpha_match'] = results['alpha_loo'].values == results['alpha_aloo'].values
    results['score_diff'] = results['score_loo'].values - results['score_aloo'].values
    print(results.to_string())
