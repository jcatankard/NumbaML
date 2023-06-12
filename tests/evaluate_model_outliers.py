from numbaml.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np


def standardize(x_train, x_test):
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train - means) / stds
    x_test = (x_test - means) / stds
    return x_train, x_test


california_housing = fetch_california_housing(as_frame=True)
X = california_housing['data']
y = california_housing['target']
print(X.head().to_string())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test = standardize(X_train, X_test)


m = Ridge(alpha=1)
m.fit(X_train, y_train)
z_scores = m.model_outliers()

preds = m.predict(X_test)
print('original mse:', round(mean_squared_error(y_test, preds), 4))


z_threshold = 4
outliers = np.abs(z_scores) > z_threshold
print('number of outliers:', z_scores[outliers].size, 'out of:', z_scores.size)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, y_train = X_train[~outliers], y_train[~outliers]
X_train, X_test = standardize(X_train, X_test)

m = Ridge(alpha=1)
m.fit(X_train, y_train)
z_scores = m.model_outliers()

preds = m.predict(X_test)
print('new mse:', round(mean_squared_error(y_test, preds), 4))
