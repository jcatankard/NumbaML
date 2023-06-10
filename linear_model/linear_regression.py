from linear_model.base_model import BaseModel
from linear_model.fit import fit


class LinearRegression(BaseModel):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        self._assign_features(x)
        x, y = self._to_numpy(x), self._to_numpy(y)
        self.coef_, self.intercept_ = fit(x, y, l2_penalty=0)
