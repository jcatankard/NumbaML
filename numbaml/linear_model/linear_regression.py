from .confidence_intervals import conf_int_parameter_method
from .base_model import _BaseModel

from numpy.typing import NDArray
import numpy as np


class LinearRegression(_BaseModel):
    """https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"""
    def __init__(self, fit_intercept: bool = True):
        """
        :param fit_intercept: Whether to calculate the intercept for this model.
            If set to false, no intercept will be used in calculations (i.e. data is expected to be centered).
        """
        super().__init__(alpha=0, fit_intercept=fit_intercept)

    def conf_int(self, sig: float = .05) -> NDArray[np.float64]:
        if self._y.shape[1] > 1:
            raise NotImplementedError('Confidence intervals are only implemented for single targets')
        return conf_int_parameter_method(self._X, self._y, self.params_, sig)

    def conf_int_dict(self, sig: float = .05) -> dict:
        conf_int = self.conf_int(sig).T
        names = (['intercept'] + self.feature_names_in_) if self.fit_intercept else self.feature_names_in_
        return {
            'feature_name': names,
            'lower_bound': conf_int[0],
            'coef': self.params_,
            'upper_bound': conf_int[1]
        }
