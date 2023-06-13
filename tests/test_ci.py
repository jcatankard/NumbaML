from numbaml.linear_model import LinearRegression
from test_utils import create_data, evaluate
import statsmodels.api as sm
import numpy as np


def test_confidence_intervals(trials: int = 1):
    """this test aims to show the confidence intervals is equivalent to statsmodels"""
    for n in range(trials):
        print(f'---------------- CHECK Confidence Intervals - TRIAL {n + 1} ----------------')
        x, y, alphas = create_data(n_alphas=0)
        sig = np.random.choice([.01, .025, .05, .1])
        print(f'significance level: {sig}')

        model = sm.OLS(y, sm.add_constant(x))
        results = model.fit()
        ci1 = results.conf_int(alpha=sig)

        m = LinearRegression()
        m.fit(x, y)
        ci2 = m.conf_int(sig)
        print('upper conf:', evaluate(ci1[:, 1], ci2[:, 1]))
        print('lower conf:', evaluate(ci1[:, 0], ci2[:, 0]))
