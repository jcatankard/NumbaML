from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.datasets import make_regression
import statsmodels.api as sm
import numpy as np
import unittest
import numbaml


class TestRegression(unittest.TestCase):

    n_tests = 5
    places = 6
    atol = 1e-06
    rtol = 1e-06
    print_ = False
    min_samples = 15
    max_samples = 10_000

    def common_tests(self, model1, model2, x, y):
        print('        test .preds')
        np.testing.assert_allclose(model1.predict(x), model2.predict(x), atol=self.atol, rtol=self.rtol)
        print('        test .coef_')
        np.testing.assert_allclose(model1.coef_, model2.coef_, atol=self.atol, rtol=self.rtol)
        print('        test .intercept_')
        self.assertAlmostEqual(model1.intercept_, model2.intercept_, places=self.places)
        print('        test .n_features_in_')
        np.testing.assert_allclose(model1.n_features_in_, model2.n_features_in_, atol=self.atol, rtol=self.rtol)
        print('        test .score')
        self.assertAlmostEqual(model1.score(x, y), model2.score(x, y), places=self.places)

    def ci_test(self, model, x, y, fit_intercept):
        print('        test .conf_int')
        x = sm.add_constant(x) if fit_intercept else x
        m3 = sm.OLS(y, x)
        r3 = m3.fit()
        sig = np.random.choice([0.01, 0.025, 0.05, 0.1], size=None)
        ci1 = model.conf_int(sig=sig, bootstrap_method=False)
        ci3 = r3.conf_int(alpha=sig)
        np.testing.assert_allclose(ci1, ci3, atol=self.atol, rtol=self.rtol)

    def alpha_test(self, model1, model2):
        print('        test .alpha_')
        self.assertEqual(model1.alpha_, model2.alpha_)

    def best_score_test(self, model1, model2):
        print('        test .best_score_')
        self.assertAlmostEqual(model1.best_score_, model2.best_score_, places=self.places)

    def select_scoring(self, cv: int) -> str:
        scoring = None if cv is None else np.random.choice(['r2', 'neg_mean_squared_error'])
        if self.print_:
            print(f'scoring = {scoring}')
        return scoring

    def select_cv(self) -> int:
        leave_one_out = np.random.choice([True, False])
        cv = np.random.randint(2, self.min_samples) if leave_one_out else None
        if self.print_:
            print(f'cv = {cv}')
        return cv

    def select_fit_intercept(self) -> bool:
        fit_intercept = np.random.choice([True, False])
        if self.print_:
            print(f'fit_intercept = {fit_intercept}')
        return fit_intercept

    def create_data(self):
        n_samples = np.random.randint(self.min_samples, self.max_samples)
        max_features = 100 if n_samples // 2 > 100 else n_samples // 2
        n_features = np.random.randint(1, max_features)
        noise = np.random.randint(10, 100)
        x, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
        if self.print_:
            print(f'n_samples = {n_samples}, n_features = {n_features}, noise = {noise}')
        return x, y

    def select_alphas(self, n_alphas: int):
        alphas = np.array([.001, .01, .1, .5, .9, .99, 1., 1.1, 2.5, 5., 10.])
        alphas = np.random.choice(alphas, n_alphas, replace=False)
        if self.print_:
            print(f'alphas = {alphas}')
        return alphas

    @staticmethod
    def fit_models(x, y, model_type1, model_type2, kwargs: dict):
        m1 = model_type1(**kwargs)
        m2 = model_type2(**kwargs)
        m1.fit(x, y)
        m2.fit(x, y)
        return m1, m2

    def test_linear_regression(self):
        print('testing LinearRegression')
        for i in range(self.n_tests):
            print(f'    test {i + 1}')
            x, y = self.create_data()
            kwargs = {'fit_intercept': self.select_fit_intercept()}
            m1, m2 = self.fit_models(x, y, numbaml.linear_model.LinearRegression, LinearRegression, kwargs=kwargs)
            self.common_tests(m1, m2, x, y)
            self.ci_test(m1, x, y, kwargs['fit_intercept'])

    def test_ridge(self):
        print('testing Ridge')
        for i in range(self.n_tests):
            print(f'    test {i + 1}')
            x, y = self.create_data()
            kwargs = {'alpha': self.select_alphas(1), 'fit_intercept': self.select_fit_intercept()}
            m1, m2 = self.fit_models(x, y, numbaml.linear_model.Ridge, Ridge, kwargs=kwargs)
            self.common_tests(m1, m2, x, y)

    def test_ridgecv(self):
        print('testing RidgeCV')
        for i in range(self.n_tests):
            print(f'    test {i + 1}')
            x, y = self.create_data()
            cv = self.select_cv()
            kwargs = {'alphas': self.select_alphas(np.random.randint(2, 10)),
                      'scoring': self.select_scoring(cv),
                      'cv': cv,
                      'fit_intercept': self.select_fit_intercept()
                      }
            m1, m2 = self.fit_models(x, y, numbaml.linear_model.RidgeCV, RidgeCV, kwargs=kwargs)

            self.common_tests(m1, m2, x, y)
            self.alpha_test(m1, m2)
            self.best_score_test(m1, m2)
