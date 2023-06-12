from numbaml.cross_validation import find_alpha, find_alpha_aloocv
from numbaml.predict import predict
from numbaml.fit import fit
import numpy.typing as npt
from typing import List
import numpy as np


class BaseModel:
    def __init__(self):
        self.features: List[str] = None
        self.coef_: npt.NDArray = None
        self.intercept_: float = None

    def predict(self, x) -> npt.NDArray:
        x = self._to_numpy(x)
        return predict(x, self.coef_, self.intercept_)

    def _assign_features(self, x):
        n_features = x.shape[1]
        self.features = x.columns if hasattr(x, 'columns') else list(map(str, range(n_features)))

    @staticmethod
    def _to_numpy(a) -> npt.NDArray:
        return a.to_numpy(dtype=np.float64) if hasattr(a, 'to_numpy') else a.astype(np.float64)

    def model_details(self) -> dict:
        m = {k: v for k, v in self.__dict__.items() if k not in ['features', 'coef_']}
        m['coef_'] = {} if self.coef_ is None else dict(zip(self.features, self.coef_))
        m['model'] = self.__class__
        return m


class LinearRegression(BaseModel):
    """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"""
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        self._assign_features(x)
        x, y = self._to_numpy(x), self._to_numpy(y)
        self.coef_, self.intercept_ = fit(x, y, l2_penalty=0)


class Ridge(BaseModel):
    """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge"""
    def __init__(self, alpha: float = 1.):
        """
        :param alpha: l2 penalization.
        """
        super().__init__()
        self.alpha_: float = float(alpha)

    def fit(self, x, y):
        self._assign_features(x)
        x, y = self._to_numpy(x), self._to_numpy(y)
        self.coef_, self.intercept_ = fit(x, y, l2_penalty=self.alpha_)


class RidgeCV(BaseModel):
    """scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV"""
    def __init__(self, alphas: List[float] = (0.1, 1.0, 10.0), cv: int = None, scoring: str = None):
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
