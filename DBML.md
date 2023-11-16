# DBML

Causal inference with de-biased / double machine using principles from Frisch–Waugh–Lovell theorem.
This implementation currently just uses Ridge regression for de-noising and de-biasing but in the future could be expanded to any regression model
(note: the theorem does not extend to classification models such as logit regression).

Causal inference is done by decomposing the regression problem into three steps:
1. de-noise: regress y on X (confounding variables) and take residuals
2. de-bias: regress treatment variable on X (confounding variables) and take residuals
3. inference: find coefficient of treatment by regressing residuals from step 1 on the residuals from step 2

 - RidgeCV is used in de-noising & de-biasing steps to reduce likelihood over-fitting,
for handling large number of features relative to samples as well as correlated features.
   - Out-of-fold residuals are also preferred to reduce likelihood over-fitting.

 - Standard OLS regression is used to find the best estimate for the treatment effect.

## References
1. A good derivation of the FWL theorem can be found
[here](https://www.hbs.edu/research-computing-services/Shared%20Documents/Training/fwlderivation.pdf).
2. This was mainly inspired by Causal Inference in Python and
[Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/Debiasing-with-Orthogonalization.html).

## Docs
### RidgeCausalInference
#### Parameters
 - x: array of shape (n_samples, n_features)
   - these indicate confounding variable that add noise
 - t: array of shape (n_samples, )
   - this refers to the treatment allocation
 - y: array of shape (n_samples, )
   - the dependent variable that we are aiming to measure the effect from the treatment
 - alpha: list, default=(0.1, 1., 10.)
   - l2 penalty for ridge regularization
 - sig_level: float, default=0.05
   - significance level for calculating confidence intervals
   - values should be >.5 & <1. e.g. .01, .025, .05, .1
 - oof_residuals: bool, default=True
   - whether residuals should be leave-one-out out-of-fold residuals or not
   - True is preferred to avoid likelihood of over-fitting
#### Attributes
 - y_residuals: (n_samples, )
   - residuals from regressing y on x
 - t_residuals: (n_samples, )
   - residuals from regressing t on x
 - treatment_coef_: float
   - the OLS estimate for the coefficient of the treatment variable on y after de-noising and de-biasing
 - confidence_interval_: array of shape (2, )
   - confidence interval for treatment coefficient based on given significance level
#### Methods
 - denoise()
   - runs de-noising step
 - debias()
   - runs de-biasing step
 - inference()
   - runs inference step
 - run_causal_pipeline()
   - runs three steps above in order

### Notes:
#### Treatment variable
The treatment variable can be binary (i.e. either on or off)
or a continuous treatment (e.g. number of emails sent to audience).
In either case, the analysis will perform best when there is some degree of randomisation in the allocation of the treatment.

#### Converting treatment effect to percentage uplift
The treatment_coef_ (and confidence_interval_) is the regression coefficient for the treatment variable.
If any logarithmic transformations of the y variable is performed then the treatment coefficient
can be re-interpreted as a percentage treatment effect. Guidance on how to do so can be found
[here](https://library.virginia.edu/data/articles/interpreting-log-transformations-in-a-linear-model).

## Example
```
from dbml.ridge_causal_inference import RidgeCausalInference


X = ...
t = ...
y = ...

rci = RidgeCausalInference(x, t, y, alphas=[0, .1, .5, .9, 1., 5. 10.], oof_residuals=False)
rci.run_causal_pipeline()

print(rci.treatment_coef_)
print(rci.confidence_intervals_)

```