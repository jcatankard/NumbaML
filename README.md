# Regression modelling with Numba

This implementation aims to recreate LinearRegression, Ridge & RidgeCV using Scikit-Learn as a benchmark to evaluate that equality of output.

The classes, built in regular Python, act as wrappers for the underlying Numba functions. This allows models to be used
in the regular way, as per Scikit-Learn, or for the underlying functions to be accessed as part of Numba workflow without having to leave the Numba ecosystem.

The aim has been to reproduce the key functionality from Scikit-Learn as accurately as possible, though many options available through Scikit-Learn are not available here.

### Blog posts
1. Study into implementation of efficient leave-one-out cross validation for RidgeCV - 
([medium.com](https://medium.com/@jcatankard_76170/efficient-leave-one-out-cross-validation-f1dee3b68dfe)).
2. Study into parameterized calculation of confidence intervals for model parameters - 
([medium.com](https://medium.com/@jcatankard_76170/linear-regression-parameter-confidence-intervals-228f0be5ea82)).

## Docs
### LinearRegression, Ridge & RidgeCV
#### Parameters
 - fit_intercept: bool, default=True
#### Attributes
 - coef_: array of shape (n_features, )
 - intercept_: float
 - params_: array of shape (n_features + 1, )
 - n_features_in_: int
 - feature_names_in_: ndarray of shape (n_features_in_,)
#### Methods
 - fit(X, y)
   - fit linear model
 - predict(X): array, shape (n_samples,)
   - predict using the linear model
 - score(X, y): float
   - return the coefficient of determination of the prediction
   - return: float
 - conf_int(sig=.05, bootstrap_method=False, bootstrap_iterations: int = 1000): array, shape (n_features + 1, 2)
   - confidence intervals for each parameter including intercept
 - model_outliers(): array, shape (n_samples,)
   - z-score for each sample used for fitting

### Ridge
#### Parameters
Above plus:
 - alpha: float, default=1.0

### RidgeCV
#### Parameters
Above plus:
 - alphas: array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
 - scoring: {'r2', 'neg_mean_squared_error'}, default=None
 - cv: int, default=None
#### Attributes
Above plus:
 - alpha_: float
 - best_score_: float
 - gcv_mode: {‘svd’, ‘eigen’}

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
2. ##### bootstrap approach
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
{'fit_intercept': True,
'n_features_in_': 10,
'feature_names_in_': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
'params_': array([-0.15692794, 94.9493805 , 72.75945281, 70.12131341, 44.14458294, 16.14937704, 10.7241133,
                  29.59714629, 27.53633061, 27.92558347, 70.10592426
                  ]),
'intercept_': -0.15692793734064825,
'alpha_': 0.1,
'alphas': array([ 0.1,  1. , 10. ]),
'cv': 140,
'scoring': 'neg_mean_squared_error',
'r2': False,
'gcv': 'svd',
'best_score_': -1.0806636242503618,
'coef_': {'0': 94.94938050087862, '1': 72.75945281096875, '2': 70.12131340912383, '3': 44.144582939396216,
          '4': 16.149377036680093, '5': 10.724113296900835, '6': 29.597146285642456, '7': 27.536330606887137,
          '8': 27.92558346643737, '9': 70.1059242637577
          },
'model': <class 'numbaml.linear_model.RidgeCV'>
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