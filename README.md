# Regression modelling with Numba

This implementation aims to recreate LinearRegression, Ridge & RidgeCV using Scikit-Learn as a benchmark to evaluate that equality of output.

The classes, built in regular Python, act as wrappers for the underlying Numba functions. This allows models to be used
in the regular way, as per Scikit-Learn, or for the underlying functions to be accessed as part of Numba workflow without having to leave the Numba ecosystem.

The aim has been to reproduce the key functionality from Scikit-Learn as accurately as possible, though many options available through Scikit-Learn are not available here.


## Example usage:

### LinearRegression

```
from numbaml.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Ridge

```
from numbaml.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=.9)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### RidgeCV

When leaving **CV=None**,
a highly efficient version of cross-validation is used replicating the implementation in Scikit-Learn.

```
from numbaml.linear_model import RidgeCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RidgeCV(alphas=[.5, .9, 1., 10.], cv=5, scoring='r2')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


### Other features
A couple of extra features having been added which may be useful.


#### conf_int
1. ##### parametric approach
Method that return confidence intervals for model parameters (intercept and coefs).
```
from numbaml.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=.9)
model.fit(X_train, y_train)
ci = model.conf_int(sig=0.05)
lower, upper = ci[:, 0], ci1[:, 1]
```
##### 2. bootstrap approach
An alternative non-parametric approach is also available.
The results should be close to the parametric version though not identical.
The higher the number of bootstrap iterations, the more stable the confidence intervals.
However increasing the order of magnitude of iterations will increase execution time.
```
from numbaml.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=.9)
model.fit(X_train, y_train)
ci = model.conf_int(sig=0.05, bootstrap_method=True, bootstrap_iterations=10 ** 5)
lower, upper = ci[:, 0], ci1[:, 1]
```


#### model_details
Using this method on a model will return a dictionary of the key model attributes
and also making it much easier to see the coefficients along with the column name (if using Pandas DataFrame).
```
model.model_details()
```
Example output:
```
{'intercept_': 0.4021965238329545,
'alphas': array([0.9 , 1.1 , 1.  , 0.01, 5.  ]),
'cv': 2,
'scoring': 'neg_mean_squared_error',
'r2': False, 
'gcv': False,
'alpha_': 0.01,
'best_score_': -118.13413373855249,
'coef_': {'feature_1': 19.765871603599187,
          'feature_2': 51.17752219575558
         },
'model': <class 'linear_model.ridge_cv.RidgeCV'>
}
```


#### model_outliers
It is possible to detect which data points in the training data have an out-sized influence on the model
by using leave-one-out cv. These datapoints may need investigating
and if necessary removed from the training data before re-fitting. 

```
from numbaml.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np


X, y = make_regression(random_state=2)

# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
m = Ridge(alpha=1)
m.fit(X_train, y_train)

# evaluate
preds = m.predict(X_test)
print('original mse:', round(mean_squared_error(y_test, preds), 4))

# flag outliers
z_scores = m.model_outliers()
z_threshold = 4
outliers = np.abs(z_scores) > z_threshold
print('number of outliers:', z_scores[outliers].size, 'out of:', z_scores.size)

# re-train
X_train, y_train = X_train[~outliers], y_train[~outliers]
m = Ridge(alpha=1)
m.fit(X_train, y_train)
z_scores = m.model_outliers()

# evaluate
preds = m.predict(X_test)
print('new mse:', round(mean_squared_error(y_test, preds), 4))

```