from linear_model.cross_validation import find_alpha, find_alpha_aloocv
from linear_model.base_model import BaseModel
from linear_model.fit import fit
from typing import List
import numpy as np


class RidgeCV(BaseModel):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV
    """
    def __init__(self, alphas: List[float] = (0.1, 1.0, 10.0), cv: int = None, scoring: str = None):
        """
        :param alphas: array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
            Regularization strength; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of the estimates.
            Larger values specify stronger regularization
        :param cv: Determines the cross-validation splitting strategy
            None, to use the efficient Leave-One-Out cross-validation
            integer, to specify the number of folds.
        :param scoring: If None, the negative mean squared error if cv is None (i.e. when using leave-one-out cross-validation)
            and r2 score otherwise.
            'r2' for (coefficient of determination) regression score
            'neg_mean_squared_error' for the negative mean squared error
        """
        super().__init__()
        self.alphas = np.array(alphas, dtype=np.float64)
        self.cv = cv
        self.scoring = scoring
        self.r2: bool = None
        self.gcv: bool = None
        self.alpha_: float = None
        self.best_score_: float = None

    def fit(self, x, y):
        self._assign_features(x)
        self._assign_cv(x.shape[0])
        self._assign_scoring(x.shape[0])

        x, y = self._to_numpy(x), self._to_numpy(y)

        self.alpha_, self.best_score_ = find_alpha_aloocv(x, y, self.alphas) if self.gcv\
            else find_alpha(x, y, self.alphas, self.cv, self.r2)

        self.coef_, self.intercept_ = fit(x, y, self.alpha_)

    def _assign_cv(self, n_samples: int):
        self.cv = n_samples if self.cv is None else self.cv
        if (self.cv <= 1) | (self.cv > n_samples):
            raise ValueError('CV must be >= 2 and <= number of samples')
        self.gcv = self.cv == n_samples

    def _assign_scoring(self, n_samples: int):
        if self.scoring is None:
            self.scoring = 'r2' if self.cv < n_samples else 'neg_mean_squared_error'
        elif (self.scoring == 'r2') & (self.cv == n_samples):
            raise ValueError('R^2 score is not well-defined with leave one out cv.')
        self.r2 = self.scoring == 'r2'
