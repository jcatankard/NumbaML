# Regression modelling with Numba

This implementation aims to recreate LinearRegression, Ridge & RidgeCV using Scikit-Learn as a benchmark to evaluate that equality of output.

The classes act as Python wrappers for the underlying Numba functions. This allows models to be used
exactly like Scikit-Learn or for the underlying functions to be accessed as part of Numba workflow without having to leave the Numba ecosystem.

The aim has been to reproduce the key functionality from Scikit-Learn as accurately as possible.

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
   - confidence intervals for each parameter (inc. intercept) including intercept
 - conf_int_dict(sig=.05, bootstrap_method=False, bootstrap_iterations: int = 1000): dict
   - returns feature names (inc. intercept) with coef values + confidence intervals in a dict that can be transformed into a dataframe
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

#### conf_int_dict
Return parameter estimates and confidence intervals as a dictionary that can easily been turned into a Pandas DataFrame.
If there are feature names seen in the X variables passed to "fit", they will output in the "feature_name" column.
```
from numbaml.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import pandas as pd


X, y = make_regression(random_state=2)

# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
m = Ridge(alpha=1)
m.fit(X_train, y_train)

param_dict = m.conf_int_dict(sig=0.05, bootstrap_method=True, bootstrap_iterations=10 ** 5)
param_df = pd.DataFrame(param_dict)
print(param_df)
```
Output example:
```
   feature_name  lower_bound       coef  upper_bound
0     intercept    -0.275087   0.010771     0.296628
1             0    30.125988  30.414877    30.703765
2             1    14.479350  14.796072    15.112795
3             2    59.733994  60.050851    60.367707
4             3    69.379268  69.654780    69.930292
5             4    86.762219  87.076998    87.391777
6             5    43.671286  43.953831    44.236375
7             6    81.288409  81.571708    81.855008
8             7    32.565543  32.881347    33.197150
9             8    22.464876  22.752157    23.039439
10            9    37.103956  37.382373    37.660790
```
#### model_outliers
It is possible to detect which data points in the training data have an out-sized influence on the model
by using leave-one-out cv. These datapoints may need investigating
and if necessary removed from the training data before re-fitting. 

```
from numbaml.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np


X, y = make_regression(random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

m = Ridge(alpha=1)
m.fit(X_train, y_train)
preds = m.predict(X_test)

# flag outliers
z_scores = m.model_outliers()
z_threshold = 4
outliers = np.abs(z_scores) > z_threshold
print('number of outliers:', z_scores[outliers].size, 'out of:', z_scores.size)

```