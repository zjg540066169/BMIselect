#' Multi-Laplace MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection under the Multi-Laplace prior on
#' regression coefficients across multiply-imputed datasets.  The prior shares
#' local shrinkage parameters (\code{lambda2}) across imputations and places
#' a Gamma(\code{h}, \code{v}) hyperprior on the global parameter \code{rho}.
#' The Gibbs sampler is implemented in C++ (RcppArmadillo).
#'
#' @param X A 3-D array of predictors with dimensions \code{D × n × p}.
#' @param Y A matrix of outcomes with dimensions \code{D × n}.
#' @param intercept Logical; include an intercept? Default \code{TRUE}.
#' @param h Numeric; shape parameter of the Gamma prior on \code{rho}. Default \code{2}.
#' @param v Numeric or \code{NULL}; scale parameter of the Gamma prior on \code{rho}.
#'   If \code{NULL}, defaults to \code{(D+1)/(D*(h-1))}.
#' @param nburn Integer; number of burn-in iterations. Default \code{4000}.
#' @param npost Integer; number of post-burn-in samples to store. Default \code{4000}.
#' @param seed Integer or \code{NULL}; random seed for reproducibility. Default \code{NULL}.
#' @param verbose Logical; print progress messages? Default \code{TRUE}.
#' @param printevery Integer; print progress every this many iterations. Default \code{1000}.
#' @param chain_index Integer; index of this MCMC chain (for messages). Default \code{1}.
#'
#' @return A named \code{list} with elements:
#' \describe{
#'   \item{\code{post_beta}}{Array \code{npost × D × p} of sampled regression coefficients.}
#'   \item{\code{post_alpha}}{Matrix \code{npost × D} of sampled intercepts (if used).}
#'   \item{\code{post_sigma2}}{Numeric vector of length \code{npost}, sampled residual variances.}
#'   \item{\code{post_lambda2}}{Matrix \code{npost × p} of sampled local shrinkage parameters.}
#'   \item{\code{post_rho}}{Numeric vector of length \code{npost}, sampled global parameters.}
#'   \item{\code{post_fitted_Y}}{Array \code{npost × D × n} of posterior predictive draws (with noise).}
#'   \item{\code{post_pool_beta}}{Matrix \code{(npost * D) × p} of pooled coefficient draws.}
#'   \item{\code{post_pool_fitted_Y}}{Matrix \code{(npost * D) × n} of pooled predictive draws (with noise).}
#'   \item{\code{hat_matrix_proj}}{Array \code{D × n × n} of averaged projection hat-matrices.}
#'   \item{\code{h}, \code{v}}{Numeric; the shape and scale hyperparameters used.}
#' }
#' @keywords internal
multi_laplace_mcmc = function(X, Y, intercept = TRUE, h = 2, v = NULL,
                              nburn = 4000, npost = 4000, seed = NULL,
                              verbose = TRUE, printevery = 1000,
                              chain_index = 1) {
  if (is.null(dim(Y))) Y = t(sapply(seq_len(dim(X)[1]), function(i) Y))
  cpp_multi_laplace(X, Y, intercept, h,
                    if (is.null(v)) NA_real_ else as.numeric(v),
                    as.integer(nburn), as.integer(npost), seed,
                    verbose, as.integer(printevery), as.integer(chain_index))
}
