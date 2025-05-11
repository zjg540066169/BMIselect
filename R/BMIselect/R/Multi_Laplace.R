# Update global shrinkage parameter rho
ML_update_rho = function(lambda2, D, p, h, s) {
  a = h + p * (D + 1) / 2
  t = 1 / (1 / s + D * sum(lambda2) / 2)
  return(rgamma(1, a, scale = t))
}

# Update local shrinkage lambda2 (per predictor)
ML_update_lambda2 = function(beta, D, rho, sigma2) {
  lambda2 = rep(0, dim(beta)[2])
  for (i in 1:dim(beta)[2]) {
    lambda2[i] = GIGrvg::rgig(1, 0.5, t(beta[, i]) %*% beta[, i] / sigma2, D * rho)
  }
  return(lambda2)
}

# Update residual variance sigma2
ML_update_sigma2 = function(X, Y, beta, alpha, lambda2) {
  D = dim(X)[1]; n = dim(X)[2]; p = dim(X)[3]
  a = D * (n + p) / 2
  SSE = sum(sapply(1:D, function(d) sum((Y[d, ] - X[d,,] %*% beta[d, ] - alpha[d])^2)))
  SSE_beta = sum(sapply(1:p, function(j) t(beta[,j]) %*% beta[,j] / lambda2[j]))
  b = (SSE + SSE_beta) / 2
  return(MCMCpack::rinvgamma(1, a, b))
}

# Update regression coefficients beta
ML_update_beta = function(X, Y, alpha, lambda2, sigma2) {
  D = dim(X)[1]; n = dim(X)[2]; p = dim(X)[3]
  beta = matrix(0, D, p)
  inv_variance = diag(1 / lambda2)
  for (d in 1:D) {
    va = Rfast::spdinv(t(X[d,,]) %*% X[d,,] + inv_variance)
    mu = (t(Y[d,] - alpha[d]) %*% X[d,,]) %*% va
    beta[d,] = MASS::mvrnorm(1, mu, va * sigma2)
  }
  return(beta)
}

# Update intercepts alpha
ML_update_alpha = function(X, Y, beta, sigma2) {
  D = dim(X)[1]; n = dim(X)[2]
  alpha = numeric(D)
  for (d in 1:D) {
    mu = mean(Y[d, ] - X[d,,] %*% beta[d, ])
    alpha[d] = rnorm(1, mu, sqrt(sigma2 / n))
  }
  return(alpha)
}

# Compute pooled residual variance and initialize beta/alpha
pooledResidualVariance <- function(M_X, M_Y) {
  m <- dim(M_X)[1]
  res_var <- numeric(m)
  beta <- matrix(0, m, dim(M_X)[3])
  alpha <- numeric(m)
  for (i in 1:m) {
    df <- as.data.frame(M_X[i,,])
    colnames(df) <- paste0("x", 1:ncol(df))
    df$y <- M_Y[i, ]
    fit <- lm(y ~ ., data = df)
    res_var[i] <- summary(fit)$sigma^2
    beta[i, ] <- fit$coefficients[-1]
    alpha[i] <- fit$coefficients[1]
  }
  pooled_res_var <- mean(res_var)
  pooled_se <- sqrt(var(res_var) / m)
  return(list(pooled_res_var = pooled_res_var,
              individual_res_vars = res_var,
              pooled_se = pooled_se,
              beta = beta,
              alpha = alpha))
}


#' Multi-Laplace MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using the Multi_Laplace prior
#' across multiply-imputed datasets. This prior induces group-wise shrinkage
#' on regression coefficients by sharing local shrinkage (lambda) across datasets
#' and placing a gamma hyperprior on the global shrinkage (rho).
#'
#' @param X A 3D array of predictors with dimensions \code{D x n x p}.
#' @param Y A matrix of outcomes with dimensions \code{D x n}.
#' @param h Shape parameter for global shrinkage prior. Default: \code{2}.
#' @param s Scale parameter for global shrinkage prior. Default: \code{(D+1)/D}.
#' @param nburn Number of burn-in iterations. Default: \code{4000}.
#' @param npost Number of posterior samples to store. Default: \code{4000}.
#' @param seed Optional random seed.
#' @param verbose Logical; whether to print progress. Default: \code{TRUE}.
#' @param printevery Integer; how often to print progress. Default: \code{1000}.
#' @param chain_index Integer; index of the current MCMC chain (used for printing).
#'
#' @return A list containing posterior draws of:
#' \describe{
#'   \item{post_beta}{\code{npost x D x p} array of regression coefficients.}
#'   \item{post_alpha}{\code{npost x D} matrix of intercepts.}
#'   \item{post_sigma2}{\code{npost} vector of residual variances.}
#'   \item{post_lambda2}{\code{npost x p} matrix of local shrinkage parameters for each predictor.}
#'   \item{post_rho}{\code{npost} vector of global shrinkage parameters.}
#'   \item{post_fitted_Y}{\code{npost x D x n} array of posterior predictive means (with noise).}
#'   \item{post_pool_beta}{\code{(npost * D) x p} matrix; reshaped from \code{post_beta} for pooling.}
#'   \item{post_pool_fitted_Y}{\code{(npost * D) x n} matrix; reshaped from \code{post_fitted_Y} for pooling.}
#'   \item{h}{Numeric scalar; shape hyperparameter used for the global shrinkage prior.}
#'   \item{s}{Numeric scalar; scale hyperparameter used for the global shrinkage prior.}
#' }
#' @export
#'
#' @importFrom stats as.formula coef dgamma dlnorm gaussian glm lm median plogis predict quantile rbeta rbinom rcauchy residuals rgamma rlnorm rnorm runif sd var vcov
#'
multi_laplace_mcmc = function(X, Y, h = 2, s = NULL, nburn = 4000, npost = 4000, seed = NULL, verbose = TRUE, printevery = 1000, chain_index = 1) {
  if (!is.null(seed)) set.seed(seed)

  D = dim(X)[1]; n = dim(X)[2]; p = dim(X)[3]
  if (is.null(s)) s = (D + 1) / D

  rho = rgamma(1, shape = h, scale = s)
  pool = pooledResidualVariance(X, Y)
  sigma2 = pool$pooled_res_var
  beta = pool$beta
  alpha = pool$alpha
  lambda2 = ML_update_lambda2(beta, D, rho, sigma2)

  post_lambda2 = matrix(NA, npost, p)
  post_alpha = matrix(NA, npost, D)
  post_rho = rep(NA, npost)
  post_sigma2 = rep(NA, npost)
  post_beta = array(NA, dim = c(npost, D, p))
  post_fitted_Y = array(NA, dim = c(npost, D, n))

  for (i in 1:(nburn + npost)) {
    if (i %% printevery == 0 && verbose)
      cat(sprintf("Chain %d: %d / %d (%s)\n", chain_index, i, nburn + npost, ifelse(i <= nburn, "burn-in", "sampling")))
    if (i == (nburn + 1) && verbose)
      cat(sprintf("Chain %d: %d / %d (sampling)\n", chain_index, i, nburn + npost))

    rho = ML_update_rho(lambda2, D, p, h, s)
    lambda2 = ML_update_lambda2(beta, D, rho, sigma2)
    beta = ML_update_beta(X, Y, alpha, lambda2, sigma2)
    alpha = ML_update_alpha(X, Y, beta, sigma2)
    sigma2 = ML_update_sigma2(X, Y, beta, alpha, lambda2)

    if (i >= nburn) {
      post_lambda2[i - nburn, ] = lambda2
      post_alpha[i - nburn, ] = alpha
      post_rho[i - nburn] = rho
      post_sigma2[i - nburn] = sigma2
      post_beta[i - nburn, ,] = beta

      for (d in 1:D) {
        post_fitted_Y[i - nburn, d, ] = X[d,,] %*% beta[d, ] + alpha[d] + rnorm(1, 0, sqrt(sigma2))
      }
    }
  }

  return(list(
    post_lambda2 = post_lambda2,
    post_alpha = post_alpha,
    post_rho = post_rho,
    post_sigma2 = post_sigma2,
    post_beta = post_beta,
    post_fitted_Y = post_fitted_Y,
    post_pool_beta = matrix(post_beta, nrow = npost * D, ncol = p),
    post_pool_fitted_Y = matrix(post_fitted_Y, nrow = npost * D, ncol = n),
    h = h,
    s = s
  ))
}
