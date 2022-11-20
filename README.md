# BMIselect
This package provides five Bayesian MI-LASSO models for Bayesian variable selection on multiply-imputed data. The inference for these five models are based on MCMC chains written in Python. They take multiply-imputed dataset as the input and output posterior samples of each parameter.

Multiple imputation(MI) is one of the major tools to deal with missing data. However, if we conduct variable selection methods on each imputed dataset separately, different sets of important variables may be obtained. It is very hard to decide which variable is important based on these different variable sets. MI-LASSO, proposed in 2013, is one of the most popular solutions to this problem. The MI-LASSO method treats the estimated regression coefficients of the same variable across all imputed datasets as a group and applies the group LASSO penalty to yield a consistent variable selection across multiple-imputed datasets. In this package, we extend MI-LASSO into Bayesian framework. To yield various selection effects, we use totally five different prior distributions as follows:
* Ridge prior (ARD prior)
* Horseshoe prior
* Laplace prior
* Spike-ridge prior
* Spike-laplace prior

For more details on the algorithm and its applications, please consult the following paper: "Variable Selection for Multiply-imputed Data: A Bayesian Framework" (Arxiv: https://arxiv.org/abs/2211.00114).

## Installation

Before install `BMIselect`, please make sure `pymc3` and `theano-pymc` has been installed. For more details, please refer to https://docs.pymc.io/en/v3/index.html.

After `pymc3` and `theano-pymc` have been successfully installed, you can install this package from pip with

`pip install bmiselect`


## Requirement
* Python >= 3.7
* pymc3 >= 3.11.5
* theano-pymc >= 1.1.2
* mkl >= 2.4.0
* mkl-service >= 2.4.0
* numpy
* matplotlib
* sklearn
* pandas
* seaborn
* arviz
* xarray
* statsmodels



## Models
This package is based on the linear regression model: <img src="https://latex.codecogs.com/gif.latex?Y=\alpha+X\beta+\epsilon" /> 

Different models impose different group-based prior distributions on <img src="https://latex.codecogs.com/gif.latex?\beta" />. 

<table>
   <tr>
      <th align="center">Type</th>
      <th align="center">Group-based Prior</th>
     <th align="center">Hyper-parameters</th>
      <th align="center">Default value</th>
   </tr>
   <tr>
      <td style="text-align:center" align="center" rowspan="4" colspan="1">Shrinkage Model</td>
      <td style="text-align:center" align="center" colspan="1" rowspan="2">Multi-Laplace</td>
      <td style="text-align:center" align="center" colspan="1">r</td>
      <td style="text-align:center" align="center" colspan="1">2</td>
   </tr>
   <tr>
    <td style="text-align:center" align="center" colspan="1">s</td>
      <td style="text-align:center"  align="center" colspan="1">15</td>
   </tr>
   <tr>
      <td style="text-align:center" align="center" colspan="1" rowspan="1">ARD(Ridge)</td>
      <td style="text-align:center" align="center" colspan="2">No hyper-parameters</td>
   </tr>
   <tr>
    <td style="text-align:center" align="center" colspan="1">Horseshoe</td>
      <td style="text-align:center" align="center" colspan="2">No hyper-parameters</td>
   </tr>
   <tr>
      <td style="text-align:center" align="center" colspan="1" rowspan="5">Discrete Mixture Model</td>
      <td style="text-align:center" align="center" rowspan="2">Spike-Ridge</td>
      <td style="text-align:center" align="center" rowspan="1">p0</td>
      <td style="text-align:center" align="center" rowspan="1">0.5</td>
   <tr>
     <td style="text-align:center" align="center" colspan="1">v0</td>
      <td style="text-align:center" align="center" colspan="1">4</td>
   </tr>
   <tr>
      <td style="text-align:center" align="center" colspan="1" rowspan="3">Spike-Laplace</td>
      <td style="text-align:center" align="center" colspan="1">lambda</td>
      <td style="text-align:center" align="center" colspan="1">6/11</td>
   </tr>
   <tr>
      <td style="text-align:center" align="center" colspan="1">a</td>
      <td style="text-align:center" align="center" colspan="1">1</td>
   </tr>
      <tr>
      <td style="text-align:center" align="center" colspan="1">b</td>
         <td style="text-align:center" align="center" colspan="1">1</td>
   </tr>
</tr>
</table>

The inference is done with posterior samples by running MCMC. 

## Usage
Examples about how to use this library are included in `bmiselect/main.py`.

### Input
After installation from pip, we can import Bayesian MI-LASSO classes in Python shell:
```
from bmiselect.models.ARD import ARD
from bmiselect.models.Horseshoe import Horseshoe
from bmiselect.models.Spike_ridge import Spike_ridge
from bmiselect.models.Laplace import Laplace
from bmiselect.models.Spike_laplace import Spike_laplace
```

### Initialization
Then we can use MI dataset to initialize the models:
```
# shrinkage models
model1 = Laplace(Y_array, X_array, standardize = True, r = 2, s = 15)
# model1 = Horseshoe(Y_array, X_array, standardize = True)
# model1 = Ridge(Y_array, X_array, standardize = True)

# discrete mixture models
model2 = Spike_laplace(Y_array, X_array, standardize = True, lambda_ = 6/11, a = 1, b = 1)
# model2 = Spike_ridge(Y_array, X_array, standardize = True, p0 = 0.5, v0 = 4)
```
Here `Y_array` is a 2-d data array for response variable, its dimension is `(n_imputations, n_samples)`. `X_array` is a 3-d data array for explanatory variables, its dimension is `(n_imputations, n_samples, n_features)`. If the parameter `standardize` is True, X_array is standardized and then used to run MCMC chains. If it is False, the original X_array is used to calculate MCMC chains. Other parameters are hyper-parameters for each model.


### Posterior Sampling
After initialization, we can use `sample` function to run MCMC chains and get posterior samples:
```
model1.sample(n_post = 1000, n_burn = 500, target_accept = 0.9, n_chain = 2, n_thread = 4, max_treedepth = 10, seed = 123)
# model2.sample(n_post = 1000, n_burn = 500, target_accept = 0.9, n_chain = 2, n_thread = 4, max_treedepth = 10, seed = 123)
```
The parameters for `sample` function are as follows:
* n_post(required): number of posterior samples for each chain.
* n_burn(required): number of burn-in samples for each chain.
* target_accept(default 0.9): target acceptance probability for NUTS.
* max_treedepth(default 10): maximum tree depth for NUTS.
* n_chain(default 1): number of parallel chains to run.
* n_thread(default 4): number of threads to run parallel computing.
* seed(default None): random seed. If seed is None, seed is equals to the current time in seconds since the Epoch.

We can use `get_posterior_samples` function to get posterior samples:
```
model1.get_posterior_samples(var_name = "beta", rescale = True)   # get posterior samples for the coefficients vector
model2.get_posterior_samples(var_name = "alpha", rescale = True)  # get posterior samples for the intercept
model2.get_posterior_samples(var_name = "g", rescale = True)      # get posterior samples for the hidden binary variables in discrete mixture models
```
Here `var_name` is the variable we want to sample for. `rescale` specifies whether to return coefficients in the original scale; if it is False, then coefficients corresponding to standardized covariates are return; if it is True, all the coefficients are rescaled to the original scale. If `standardize = False` in initialization stage, `rescale` has no effect. For MI data, we simply mixed up the posterior samples for each grouped coefficient among all MI sets. So the dimension of posterior samples for coefficients vector `beta` is `(n_chains, n_imputations * n_samples, n_features)`. And the dimension of intercept `alpha` is `(n_chains, n_imputations * n_samples)`.

### Summary Statistics
Our library provides a `summary` function to generate summary statistics for all the variables in the hierachical model:
```
summary_stats1 = model1.summary(rescale = True)
print(summary_stats1)
```
Here `rescale` is the same as it in function `get_posterior_samples`.


### Variable Selection
Users can use `select` function to select important variables:
```
select1 = model1.select(value = 0.95, rescale = True) # Credible Interval Criterion for Shrinkage Models
select2 = model2.select(value = 0.5,  rescale = True) # Cutting-off point for Discrete Mixture Models
```
The meaning of `value` depends on the type of models. For shrinkage models, `value` is the credible interval criterion for selection. For discrete mixture models, `value` stands for the cutting-off point for selection. For more details, please consult Chapter 3.2 in the paper: "Variable Selection for Multiply-imputed Data: A Bayesian Framework" (Arxiv: https://arxiv.org/abs/2211.00114).


### Evaluation
There are some evaluation functions in this library:
```
from bmiselect.utils.evaluation import sensitivity, specificity, f1_score, mse

sensitivity(select = select1, truth = truth)                                           # sensitivity
specificity(select = select2, truth = truth)                                           # specificity
f1_score(select = select2, truth = truth)                                              # f1 score
mse(beta, covariance, select = select1, X = X_array, Y = Y_array, intercept = True)    # mse, given coefficients and covariance matrix of ground truth
```
Here `select` and `truth` are binary vectors with length `(n_features)`. `select[i] = True` means i-th variable is selected.


### Refitting Linear Regression
After we complete the variable selection by Bayesian MI-LASSO, users can apply `fit_lr` to fit ordinary linear regression separately on each imputed dataset. Alternatively, users can utilize `pooled_coefficients` and `pooled_covariance` to directly get pooled coefficients and covariance matrix with Rubin`s Rule.
```
from bmiselect.utils.evaluation import fit_lr, pooled_coefficients, pooled_covariance

# refit linear regression by using selected variabels
lr_models = fit_lr(select1, X_array, Y_array, intercept = True))
for lr in lr_models:
    print(lr.summary())

# get pooled coefficients estimates by using Rubin`s rule
lr_coef = pooled_coefficients(select2, X_array, Y_array, intercept = True))
print(lr_coef)

# get pooled covariance estimates by using Rubin`s rule
lr_covariance = pooled_covariance(select2, X_array, Y_array, intercept = True))
print(lr_covariance)
```
If `intercept = True`, then an intercept is added to each ordinary linear regression respecitively. If `intercept = False`, then no intercept is used in linear regression.


## Disclaimer

If you find there is any bug, please contact me: jungang.zou@gmail.com.
