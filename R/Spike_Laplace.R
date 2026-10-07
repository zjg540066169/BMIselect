#' Spike-and-Laplace MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using a spike-and-slab prior with a
#' Laplace (double-exponential) slab on nonzero coefficients. Latent inclusion
#' indicators \code{gamma} follow Bernoulli(\code{theta}), and their
#' probabilities follow independent Beta(\code{a}, \code{b}) priors. A
#' partially-collapsed Gibbs sampler is used, implemented in C++
#' (RcppArmadillo).
#'
#' @param X A 3-D array of predictors with dimensions \code{D * n * p}.
#' @param Y A matrix of outcomes with dimensions \code{D * n}.
#' @param intercept Logical; include an intercept term? Default \code{TRUE}.
#' @param a Numeric; shape parameter of the Gamma prior. Default \code{2}.
#' @param b Numeric or \code{NULL}; scale parameter of the Gamma prior. If \code{NULL},
#'   defaults to \code{0.5*(D+1)/(D*(a-1))}.
#' @param nburn Integer; number of burn-in MCMC iterations. Default \code{4000}.
#' @param npost Integer; number of post-burn-in samples to retain. Default \code{4000}.
#' @param seed Integer or \code{NULL}; random seed for reproducibility. Default \code{NULL}.
#' @param verbose Logical; print progress messages? Default \code{TRUE}.
#' @param printevery Integer; print progress every this many iterations. Default \code{1000}.
#' @param chain_index Integer; index of this MCMC chain (for labeling messages). Default \code{1}.
#'
#' @return A named list with components:
#' \describe{
#'   \item{\code{post_rho}}{Numeric vector length \code{npost}, sampled global scale \eqn{\rho}.}
#'   \item{\code{post_gamma}}{Matrix \code{npost * p} of sampled inclusion indicators.}
#'   \item{\code{post_theta}}{Matrix \code{npost * p} of sampled Beta parameters \eqn{\theta_j}.}
#'   \item{\code{post_alpha}}{Matrix \code{npost * D} of sampled intercepts (if used).}
#'   \item{\code{post_lambda2}}{Matrix \code{npost * p} of sampled local scale parameters \eqn{\lambda_j^2}.}
#'   \item{\code{post_sigma2}}{Numeric vector length \code{npost}, sampled residual variances.}
#'   \item{\code{post_beta}}{Array \code{npost * D * p} of sampled regression coefficients.}
#'   \item{\code{post_fitted_Y}}{Array \code{npost * D * n} of posterior predictive draws (including noise).}
#'   \item{\code{post_pool_beta}}{Matrix \code{(npost * D) * p} of pooled coefficient draws.}
#'   \item{\code{post_pool_fitted_Y}}{Matrix \code{(npost * D) * n} of pooled predictive draws (with noise).}
#'   \item{\code{hat_matrix_proj}}{Array \code{D * n * n} of averaged projection hat-matrices.}
#'   \item{\code{a}, \code{b}}{Numeric values of the rho hyperparameters used.}
#' }
#'
#' @note
#' The partially-collapsed Gibbs update of the inclusion indicators uses a
#' rank-1 Woodbury / matrix-determinant update, ported from the reference R
#' implementation by Jungang Zou, 2024-2026.
#' @keywords internal
spike_laplace_partially_mcmc = function(X, Y, intercept = TRUE, a = 2, b = NULL,
                                        nburn = 4000, npost = 4000, seed = NULL,
                                        verbose = TRUE, printevery = 1000,
                                        chain_index = 1) {
  if (is.null(dim(Y))) Y = t(sapply(seq_len(dim(X)[1]), function(i) Y))
  cpp_spike_laplace(X, Y, intercept, a,
                    if (is.null(b)) NA_real_ else as.numeric(b),
                    as.integer(nburn), as.integer(npost), seed,
                    verbose, as.integer(printevery), as.integer(chain_index))
}
