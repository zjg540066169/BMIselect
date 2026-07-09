# BMIselect

<!-- badges: start -->
<!-- badges: end -->

**Bayesian MI-LASSO for Variable Selection on Multiply-Imputed Data.**

`BMIselect` provides a suite of Bayesian MI-LASSO methods for variable selection
on multiply-imputed datasets. It implements four Bayesian MI-LASSO models — three
shrinkage priors (**Multi-Laplace**, **Horseshoe**, **ARD**) and one
spike-and-slab prior (**Spike-and-Laplace**) — together with tools for model
fitting via MCMC, a four-step projection-predictive variable-selection procedure,
and hyperparameter calibration. The methods handle both continuous and binary
covariates under MAR or MCAR missingness. The frequentist MI-LASSO of Chen and
Wang (2013) is also provided via `MI_LASSO()`.

The MCMC samplers are implemented in C++ (via `Rcpp`/`RcppArmadillo`) for speed.

## Installation

From CRAN (once available):

```r
install.packages("BMIselect")
```

Development version from GitHub:

```r
# install.packages("devtools")
devtools::install_github("zjg540066169/BMIselect")
```

## Quick start

```r
library(BMIselect)

# simulate a multiply-imputed dataset (D = 5 imputations)
sim <- sim_A(n = 100, p = 20, type = "MAR", SNP = 1.5,
             low_missing = TRUE, n_imp = 5, seed = 123)
X <- sim$data_MI$X   # D x n x p array of imputed covariates
Y <- sim$data_MI$Y   # D x n matrix of outcomes

# fit a Bayesian MI-LASSO model and run the four-step selection
fit <- BMI_LASSO(X, Y, model = "Horseshoe",
                 nburn = 4000, npost = 4000, nchains = 1, ncores = 1)

fit$best_select   # selected variables
```

Available `model` values: `"Multi_Laplace"`, `"Horseshoe"`, `"ARD"`,
`"Spike_Laplace"`. See the package vignette (`vignette("Introduction",
package = "BMIselect")`) for a full walkthrough of fitting, projection-predictive
selection, and posterior inference.

## Citation

Zou, J., Wang, S. and Chen, Q. (2025). *Bayesian MI-LASSO for Variable Selection
on Multiply-Imputed Data.* arXiv:2211.00114.
<https://doi.org/10.48550/arXiv.2211.00114>

The frequentist MI-LASSO is from Chen, Q. and Wang, S. (2013). *Variable
selection for multiply-imputed data with application to dioxin exposure study.*
Statistics in Medicine, 32(21): 3646–3659.

## License

Apache License (>= 2). See [LICENSE](LICENSE).
