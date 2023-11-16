from .model_selection import find_alpha_kfolds, find_alpha_loo
from .base_model import _BaseModel
from .fit import fit

from typing import Optional
import numpy as np


class Ridge(_BaseModel):
    """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge"""

    def __init__(self, alpha: float = 1., fit_intercept: bool = True):
        """
        :param alpha: l2 penalization.
        :param fit_intercept: Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (i.e. data is expected to be centered).
        """
        super().__init__(alpha=float(alpha), fit_intercept=fit_intercept)


class RidgeCV(_BaseModel):
    """scikit-learn.org/stable/modules/generated/sklearn.numbaml.RidgeCV.html#sklearn.numbaml.RidgeCV"""

    def __init__(self, alphas: list[float] = (0.1, 1.0, 10.0), cv: Optional[int] = None, scoring: Optional[str] = None,
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
        self.r2: Optional[bool] = None
        self.gcv: Optional[str] = None
        self.best_score_: Optional[float] = None

    def fit(self, x, y):
        self._assign_feature_names(x)
        self._assign_cv(x.shape[0])
        self._assign_scoring(x.shape[0])

        self._X, self._y = self._to_numpy(x), self._to_numpy(y).reshape(x.shape[0], -1)
        self._X = self._add_intercept(self._X)

        if self.gcv == 'svd':
            self.alpha_, self.best_score_ = find_alpha_loo(self._X, self._y, self.alphas, self.fit_intercept)
        else:
            self.alpha_, self.best_score_ = find_alpha_kfolds(self._X, self._y, self.alphas, self.fit_intercept,
                                                              self.cv, self.r2)

        self.params_ = fit(self._X, self._y, self.alpha_, self.fit_intercept)
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
