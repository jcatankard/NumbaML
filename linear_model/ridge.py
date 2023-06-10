from linear_model.base_model import BaseModel
from linear_model.fit import fit


class Ridge(BaseModel):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
    """

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
