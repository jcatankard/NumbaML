# Regression modelling with Numba

This implementation aims to recreate LinearRegression, Ridge & RidgeCV using Scikit-Learn as a benchmark to evaluate that equality of output.

The classes, built in regular Python, act as wrappers for the underlying Numba functions. This allows models to be used
in the regular way, as per Scikit-Learn, or for the underlying functions to be accessed as part of Numba workflow without having to leave the Numba ecosystem.

The aim has been to reproduce the key functionality from Scikit-Learn as accurately as possible, though many options available through Scikit-Learn are not available here.


## Example usage:

### LinearRegression

```
from linear_model.linear_regression import LinearRegression
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
from linear_model.ridge import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=.9)
model.fit(X, y)
y_pred = model.predict(X)
```

### RidgeCV

When leaving **CV=None**,
a highly efficient version of cross-validation is used replicating the implementation in Scikit-Learn.

```
from linear_model.ridge import RidgeCV
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