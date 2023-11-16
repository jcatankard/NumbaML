from ..linear_model.model_selection import approximate_leave_one_out_errors
from ..linear_model import RidgeCV, LinearRegression

from numpy.typing import NDArray
from typing import Optional
import numpy as np


class RidgeCausalInference:
    """
    Causal inference with de-biased / double machine using principles from Frisch–Waugh–Lovell theorem

    Causal inference is done by decomposing the regression problem into three steps:
        1. de-noise: regress y on X (confounding variables) and take residuals
        2. de-bias: regress treatment variable on X (confounding variables) and take residuals
        3. inference: find coefficient of treatment by regressing residuals from step 1 on the residuals from step 2

    RidgeCV is used in de-noising & de-biasing steps to reduce likelihood over-fitting,
    for handling large number of features relative to samples as well as correlated features.
    Out-of-fold residuals are also preferred to reduce likelihood over-fitting.
    Standard OLS regression is used to find the best estimate for the treatment effect.
    """

    def __init__(self, x, t, y,
                 alphas: list[float] = (0.1, 1.0, 10.0),
                 sig_level: float = 0.05,
                 out_of_fold_residuals: bool = True
                 ):
        """
        :param x: matrix of confounding features
        :param t: vector indicating treatment which we want to measure
        :param y: target variable
        :param sig_level: probability of a type-ii error for calculating confidence intervals
        :param out_of_fold_residuals: whether to use out-of-fold residuals (highly advised) or standard residuals
        """
        self.alphas = alphas
        self.sig_level = sig_level
        self.out_of_fold_residuals = out_of_fold_residuals

        self._X, self._t, self._y = self._to_numpy(x), self._to_numpy(t), self._to_numpy(y)

        self.t_residuals: Optional[NDArray[np.float64]] = None
        self.y_residuals: Optional[NDArray[np.float64]] = None
        self.treatment_coef_: Optional[float] = None
        self.confidence_interval_: Optional[NDArray[np.float64]] = None

    @staticmethod
    def _to_numpy(a) -> NDArray[np.float64]:
        a = np.asarray(a, dtype=np.float64, order='C')
        return a.reshape(a.shape[0], -1)

    def _residuals(self, target: NDArray[np.float64]) -> NDArray[np.float64]:
        m = RidgeCV(self.alphas)
        m.fit(self._X, target)
        if self.out_of_fold_residuals:
            res = approximate_leave_one_out_errors(m._X, target, m.alpha_, fit_intercept=True)
            return res.reshape(target.shape[0], -1)
        else:
            pred = m.predict(self._X).reshape(target.shape[0], -1)
            return target - pred

    def denoise(self):
        self.y_residuals = self._residuals(self._y)

    def debias(self):
        self.t_residuals = self._residuals(self._t)

    def inference(self):
        m = LinearRegression(fit_intercept=False)
        m.fit(self.t_residuals, self.y_residuals)
        self.treatment_coef_ = m.coef_
        self.confidence_interval_ = m.conf_int(sig=self.sig_level)

    def run_causal_pipeline(self):
        self.denoise()
        self.debias()
        self.inference()
