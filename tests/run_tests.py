from tests.test_functions import test_linear_regression, test_ridge, test_ridgecv, test_aloocv, test_fit_intercept_false
from tests.test_ci import test_confidence_intervals


test_linear_regression(3)
test_ridge(3)
test_ridgecv(3)
test_aloocv(3)
test_fit_intercept_false(3)
test_confidence_intervals(3)
