from numbaml.model_selection import find_alpha_kfolds, find_alpha_loo, approximate_leave_one_out_errors, r2_score
from numbaml.confidence_intervals import conf_int_parameter_method, conf_int_bootstrap_method
from numbaml.predict import predict
from numbaml.fit import fit
import numpy.typing as npt
from typing import List
import numpy as np


class LinearRegression:
    """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"""
    def __init__(self, fit_intercept: bool = True):
        """
        :param fit_intercept: Whether to calculate the intercept for this model.
        If set to false, no intercept will be used in calculations (i.e. data is expected to be centered).
        """
        self.fit_intercept = fit_intercept
        self.n_features_in_: int = None
        self.feature_names_in_: List[str] = None
        self.coef_: npt.NDArray = None
        self.params_: npt.NDArray = None
        self.intercept_: float = None
        self.alpha_: float = float(0)
        self._X: npt.NDArray = None
        self._y: npt.NDArray = None

    def fit(self, x, y):
        self._assign_feature_names(x)
        self._X, self._y = self._to_numpy(x), self._to_numpy(y)
        self._X = self._add_intercept(self._X)
        self.params_ = fit(self._X, self._y, l2_penalty=self.alpha_)
        self._assign_params()

    def predict(self, x) -> npt.NDArray:
        x = self._to_numpy(x)
        x = self._add_intercept(x)
        return predict(x, self.params_)

    def _add_intercept(self, x: npt.NDArray) -> npt.NDArray:
        if self.fit_intercept:
            n_samples = x.shape[0]
            intercept = np.ones((n_samples, 1), dtype=np.float64)
            x = np.concatenate([intercept, x], axis=1)
        return x

    def _assign_params(self):
        self.intercept_ = self.params_[0] if self.fit_intercept else float(0)
        self.coef_ = self.params_[1:] if self.fit_intercept else self.params_

    def _assign_feature_names(self, x):
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = x.columns if hasattr(x, 'columns') else list(map(str, range(self.n_features_in_)))

    @staticmethod
    def _to_numpy(a) -> npt.NDArray:
        return np.asarray(a, dtype=np.float64, order='C')

    def model_outliers(self) -> npt.NDArray:
        """
        calculate error z-scores to determine which datapoints have an out-sized influence on model performance
        https://hackernoon.com/how-to-use-approximate-leave-one-out-cross-validation-to-build-better-models-vg1u35g2/
        """
        errors = approximate_leave_one_out_errors(self._X, self._y, self.alpha_)
        mean, stdev = np.mean(errors), np.std(errors)
        return (errors - mean) / stdev

    def conf_int(self, sig=.05, bootstrap_method=False, bootstrap_iterations: int = 1000):
        if bootstrap_method:
            return conf_int_bootstrap_method(self._X, self._y, self.alpha_, sig, bootstrap_iterations)
        else:
            return conf_int_parameter_method(self._X, self._y, self.params_, alpha=sig)

    def conf_int_dict(self, sig=.05, bootstrap_method=False, bootstrap_iterations: int = 1000) -> dict:
        conf_int = self.conf_int(sig, bootstrap_method, bootstrap_iterations).T
        return {
            'feature_name': ['intercept'] + self.feature_names_in_,
            'lower_bound': conf_int[0],
            'coef': self.params_,
            'upper_bound': conf_int[1]
        }

    def score(self, x, y):
        """
        :param x: test sample
        :param y: true values for x
        :return: coefficient of determination
        """
        x = self._to_numpy(x)
        x = self._add_intercept(x)
        y_pred = predict(x, self.params_)
        return r2_score(y, y_pred)


class Ridge(LinearRegression):
    """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge"""

    def __init__(self, alpha: float = 1., fit_intercept: bool = True):
        """
        :param alpha: l2 penalization.
        :param fit_intercept: Whether to calculate the intercept for this model.
        If set to false, no intercept will be used in calculations (i.e. data is expected to be centered).
        """
        super().__init__(fit_intercept=fit_intercept)
        self.alpha_: float = float(alpha)


class RidgeCV(LinearRegression):
    """scikit-learn.org/stable/modules/generated/sklearn.numbaml.RidgeCV.html#sklearn.numbaml.RidgeCV"""

    def __init__(self, alphas: List[float] = (0.1, 1.0, 10.0), cv: int = None, scoring: str = None,
                 fit_intercept: bool = True):
        """
        :param alphas: array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
            Regularization strength; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of the estimates.
            Larger values specify stronger regularization
        :param cv: Determines the cross-validation splitting strategy
            None, to use the efficient Leave-One-Out cross-validation
            integer, to specify the number of folds.
        :param scoring: If None, the negative mean squared error if cv is None
        (i.e. when using leave-one-out cross-validation) and r2 score otherwise.
            'r2' for (coefficient of determination) regression score
            'neg_mean_squared_error' for the negative mean squared error
        :param fit_intercept: Whether to calculate the intercept for this model.
        If set to false, no intercept will be used in calculations (i.e. data is expected to be centered).
        """
        super().__init__(fit_intercept=fit_intercept)
        self.alphas = np.array(alphas, dtype=np.float64)
        self.cv = cv
        self.scoring = scoring
        self.r2: bool = None
        self.gcv: str = None
        self.alpha_: float = None
        self.best_score_: float = None

    def fit(self, x, y):
        self._assign_feature_names(x)
        self._assign_cv(x.shape[0])
        self._assign_scoring(x.shape[0])

        self._X, self._y = self._to_numpy(x), self._to_numpy(y)
        self._X = self._add_intercept(self._X)

        self.alpha_, self.best_score_ = find_alpha_loo(self._X, self._y, self.alphas) if self.gcv == 'svd' \
            else find_alpha_kfolds(self._X, self._y, self.alphas, self.cv, self.r2)

        self.params_ = fit(self._X, self._y, self.alpha_)
        self._assign_params()

    def _assign_cv(self, n_samples: int):
        self.cv = n_samples if self.cv is None else self.cv
        if (self.cv <= 1) | (self.cv > n_samples):
            raise ValueError('CV must be >= 2 and <= number of samples')
        self.gcv = 'svd' if (self.cv == n_samples) & (n_samples > self.n_features_in_) else 'eigen'

    def _assign_scoring(self, n_samples: int):
        if self.scoring is None:
            self.scoring = 'r2' if self.cv < n_samples else 'neg_mean_squared_error'
        elif (self.scoring == 'r2') & (self.cv == n_samples):
            raise ValueError('R^2 score is not well-defined with leave one out cv.')
        self.r2 = self.scoring == 'r2'
