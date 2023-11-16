from ..metrics import r2_score
from .predict import predict
from .fit import fit

from numpy.typing import NDArray
from typing import Optional
import numpy as np



class _BaseModel:
    """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"""
    def __init__(self, alpha: Optional[float] = None, fit_intercept: bool = True):
        """
        :param fit_intercept: Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (i.e. data is expected to be centered).
        """
        self.fit_intercept = fit_intercept
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[list[str]] = None
        self.coef_: Optional[NDArray] = None
        self.params_: Optional[NDArray] = None
        self.intercept_: Optional[float] = None
        self.alpha_: Optional[float] = alpha
        self._X: Optional[NDArray] = None
        self._y: Optional[NDArray] = None

    def fit(self, x, y):
        self._assign_feature_names(x)
        self._X, self._y = self._to_numpy(x), self._to_numpy(y).reshape(x.shape[0], -1)
        self._X = self._add_intercept(self._X)
        self.params_ = fit(self._X, self._y, l2_penalty=self.alpha_, fit_intercept=self.fit_intercept)
        self._assign_params()

    def predict(self, x) -> NDArray[np.float64]:
        x = self._to_numpy(x)
        x = self._add_intercept(x)
        preds = predict(x, self.params_)
        return preds.flatten() if preds.shape[1] == 1 else preds

    def _add_intercept(self, x: NDArray) -> NDArray[np.float64]:
        if self.fit_intercept:
            n_samples = x.shape[0]
            intercept = np.ones((n_samples, 1), dtype=np.float64)
            x = np.concatenate([intercept, x], axis=1)
        return x

    def _assign_params(self):
        self.intercept_ = self.params_[0] if self.fit_intercept else np.float64(0)
        coef_ = self.params_[1:] if self.fit_intercept else self.params_
        self.coef_ = coef_.flatten() if coef_.shape[1] == 1 else coef_.T

    def _assign_feature_names(self, x):
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = list(x.columns) if hasattr(x, 'columns') else list(map(str, range(self.n_features_in_)))

    @staticmethod
    def _to_numpy(a) -> NDArray[np.float64]:
        return np.asarray(a, dtype=np.float64, order='C')

    def score(self, x, y) -> float:
        """
        :param x: test sample
        :param y: true values for x
        :return: coefficient of determination
        """
        y_pred = self.predict(x)
        return np.float64(r2_score(y.flatten(), y_pred.flatten()))
