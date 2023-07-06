from sklearn.linear_model import LinearRegression as SKLLinearRegression
from numbaml.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np


n_samples = np.random.randint(100, 250)
n_features = np.random.randint(10, 11)
noise = np.random.randint(1, 10)
X, y = make_regression(n_samples=n_samples,
                       n_features=n_features,
                       noise=noise
                       )

m = LinearRegression()
m.fit(X, y)
print(m.score(X, y))

m2 = SKLLinearRegression()
m2.fit(X, y)
print(m2.score(X, y))
