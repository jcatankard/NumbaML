from linear_model.predict import predict
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
