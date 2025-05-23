% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bmiselect.R
\name{bmiselect}
\alias{bmiselect}
\title{Bayesian MI Variable Selection via MCMC}
\usage{
bmiselect(
  X,
  Y,
  model,
  standardize = FALSE,
  shrinkage_sym_CI = seq(0.05, 0.95, by = 0.05),
  spike_slab_threshold = seq(0.05, 0.95, by = 0.05),
  bic_select = TRUE,
  nburn = 4000,
  npost = 4000,
  seed = NULL,
  nchain = 1,
  ncores = 1,
  verbose = T,
  printevery = 1000,
  ...
)
}
\arguments{
\item{X}{A \code{D x n x p} array or \code{n x p} matrix of predictors.
If \code{X} is a matrix, it is recycled into an array with \code{D = 1}.}

\item{Y}{A \code{D x n} matrix or vector of outcomes. If \code{Y} is a vector/matrix
of length \code{n}, it is recycled with \code{D = 1}.}

\item{model}{Character. Which model to run; one of
\code{"Multi_Laplace"}, \code{"Horseshoe"}, \code{"ARD"},
\code{"Spike_Normal"}, or \code{"Spike_Laplace"}.}

\item{standardize}{Logical; if \code{TRUE}, center & scale each column of \code{X}
within each dataset before MCMC.}

\item{shrinkage_sym_CI}{Numeric vector of credible-interval levels (e.g.
\code{seq(0.05,0.95,by=0.05)}) for shrinkage-based models. Only used when \code{model}
is one of \code{"Multi_Laplace"}, \code{"Horseshoe"}, \code{"ARD"}.}

\item{spike_slab_threshold}{Numeric vector of posterior-median thresholds for
spike-and-slab models. Only used when \code{model}
is either \code{"Spike_Normal"} or \code{"Spike_Laplace"}.}

\item{bic_select}{Logical; if \code{TRUE}, perform BIC-based selection for
the MCMC draws.}

\item{nburn}{Integer number of burn-in iterations.}

\item{npost}{Integer number of posterior draws to retain after burn-in.}

\item{seed}{Optional integer seed or \code{NULL} for reproducible chains.}

\item{nchain}{Integer number of independent MCMC chains.}

\item{ncores}{Integer number of parallel cores; if \code{1}, runs sequentially.}

\item{verbose}{Logical; whether to print progress messages.}

\item{printevery}{Integer; print progress every \code{printevery} MCMC iterations.}

\item{...}{Additional model-specific arguments:
\describe{
\item{h}{default \code{2}; shrinkage parameter for \code{Multi_Laplace}}
\item{s}{default \code{(D + 1)/D}; scale hyperparameter for \code{Multi_Laplace}}
\item{v02}{default \code{2}; slab variance for \code{Spike_Normal}}
\item{p0}{default \code{0.5}; prior inclusion probability for \code{Spike_Normal}}
\item{a}{default \code{2}; shape parameter for \code{Spike_Laplace}}
\item{b}{default \code{2 * (D + 1)/D}; scale parameter for \code{Spike_Laplace}}
\item{prop_sd}{default \code{1}; SD of proposal lognormal distribution in M-H step for \code{Spike_Laplace}}
}}
}
\value{
A list with components:
\describe{
\item{posterior}{
If \code{nchain = 1}, a list of posterior draws of parameters (vectors or matrices);
if \code{nchain > 1}, a list of such lists, one per chain.
}
\item{select}{
Logical inclusion indicators: if \code{nchain = 1}, a matrix of TRUE/FALSE values for each criterion and variable;
if \code{nchain > 1}, a list of such matrices, one per chain.
}
\item{best_select}{
When \code{bic_select = TRUE}, the variable selection at the best BIC threshold:
if \code{nchain = 1}, a logical vector;
if \code{nchain > 1}, a list of logical vectors, one per chain.
}
}
}
\description{
\code{bmiselect} implements Bayesian variable selection across multiply-imputed datasets
using either shrinkage-based priors or spike-and-slab priors. You can choose among
five models and obtain posterior draws, and (optionally) BIC-based best model selection.
}
\examples{
# simulate 5 imputed datasets, n = 100, p = 10
X <- array(rnorm(5 * 100 * 10), dim = c(5, 100, 10))
Y <- matrix(rnorm(5 * 100), nrow = 5, ncol = 100)

# run two chains in parallel
out <- bmiselect(
  X, Y,
  model = "Horseshoe",
  standardize = TRUE,
  shrinkage_sym_CI = seq(0.05, 0.95, by = 0.05),
  bic_select = TRUE,
  nburn = 1000,
  npost = 1000,
  nchain = 2,
  ncores = 1,
  seed = 1,
  printevery = 200
)
str(out)

}
