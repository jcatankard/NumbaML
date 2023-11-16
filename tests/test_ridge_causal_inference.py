from numbaml.dbml import RidgeCausalInference
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from numpy.typing import NDArray
import numpy as np
import unittest


class TestRegression(unittest.TestCase):

    n_tests = 5
    atol = 1e-06
    rtol = 1e-06
    min_samples = 15
    max_samples = 10_000

    def create_data(self, n_treatments: int = 1) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        n_samples = np.random.randint(self.min_samples, self.max_samples)
        max_features = 100 if n_samples // 2 > 100 else n_samples // 2
        n_features = np.random.randint(1, max_features)
        noise = np.random.randint(10, 100)

        x, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
        t = np.random.randint(0, 2, n_samples * n_treatments).reshape(n_samples, n_treatments)
        random_noise = np.random.normal(1, .1, n_samples * n_treatments).reshape(n_samples, n_treatments)
        y += (random_noise * t).sum(axis=1)

        return x, t, y

    def run_test(self, x, t, y):
        rci = RidgeCausalInference(x, t, y, alphas=[0], out_of_fold_residuals=False)
        rci.run_causal_pipeline()

        m = LinearRegression()
        m.fit(np.concatenate((x, t), axis=1), y)

        np.testing.assert_allclose(rci.treatment_coef_, m.coef_[-t.shape[1]:], atol=self.atol, rtol=self.rtol)

    def test_fwl_theorem(self):
        """alpha=0 and out-of-fold residuals oof=False should yield the same result according to FWL as OLS"""
        print('testing FWL')
        for i in range(self.n_tests):
            print(f'    test {i + 1}')
            x, t, y = self.create_data()
            self.run_test(x, t, y)

    def test_multiple_test_treatments(self):
        """alpha=0 and out-of-fold residuals oof=False should yield the same result according to FWL as OLS"""
        print('testing multiple test treatments')
        for i in range(self.n_tests):
            print(f'    test {i + 1}')

            x, t, y = self.create_data(n_treatments=np.random.randint(2, 10))
            self.run_test(x, t, y)
