# BMIselect
This package provides five Bayesian MI-LASSO models for variable selection on multiply-imputed data. The inference for these five models are based on MCMC chains written in Python. They take multiply-imputed dataset as the input and output posterior samples of each parameter.

As we all know, multiple imputation(MI) is one of the major tools to deal with missing data. It uses some imputation methods to impute multiple datasets. Then statistical analysis is conducted on each seperate dataset. Finally a pooling method such as Rubin\`s Rule will be applied to combine the results from each imputed dataset. However, if we conduct variable selection methods on each imputed dataset separately, different sets of important variables may be obtained. It is very hard to decide which variable is important based on these different variable sets. MI-LASSO, proposed in 2013, is one of the most popular solutions to this problem. The MI-LASSO method treats the estimated regression coefficients of the same variable across all imputed datasets as a group and applies the group LASSO penalty to yield a consistent variable selection across multiple-imputed datasets. In this package, we extend MI-LASSO into Bayesian framework.

## Installation

You can install this package from pip with

`pip install bmiselect`


## Requirement
* Python >= 3.7
* pymc3 >= 3.11.5
* theano-pymc >= 1.1.2
* mkl >= 2.4.0
* mkl-service >= 2.4.0
* numpy >= 1.19.2
* matplotlib >= 3.3.2
* sklearn >= 0.23.2
* pandas >= 1.1.3
