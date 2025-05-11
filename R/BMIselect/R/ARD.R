ARD_update_psi2 = function(beta, D, sigma2, eps = 1e-6){
  psi2 = rep(0, dim(beta)[2])
  for (j in 1:dim(beta)[2]) {
    psi2[j] = rgamma(1, D / 2, scale = 2 * sigma2 / (t(beta[,j]) %*% beta[,j]))
  }
  psi2 = pmin(1/eps, psi2)
  return(psi2)
}

ARD_update_sigma2 = function(X, Y, beta, alpha, psi2){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  a =  D * (n + p) / 2
  SSE = 0
  SSE_beta = 0
  for(d in 1:D){
    y_hat = as.matrix(X[d,,]) %*% beta[d,] + alpha[d]
    SSE = SSE + sum((Y[d,] - y_hat)^2)
  }

  for(j in 1:p){
    SSE_beta = SSE_beta + t(beta[,j]) %*% beta[,j] * psi2[j]
  }

  b = (SSE + SSE_beta) / 2
  return(MCMCpack::rinvgamma(1, a, b))
}

ARD_update_beta = function(X, Y, alpha, psi2, sigma2){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  beta = matrix(0, nrow = D, ncol = p)

  inv_variance = diag(psi2)
  for(d in 1:D){
    va = Rfast::spdinv((t(X[d,,]) %*% as.matrix(X[d,,]) + inv_variance))
    mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X[d,,])) %*% va
    va = va * sigma2
    beta[d,] = MASS::mvrnorm(1, mu, va)
  }
  return(beta)
}


ARD_update_alpha = function(X, Y, beta, sigma2){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  alpha = rep(0, D)
  for(d in 1:D){
    va = sigma2 / n
    mu = mean(Y[d,] - as.matrix(X[d,,]) %*% beta[d,])
    alpha[d] = rnorm(1, mu, sqrt(va))
  }
  return(alpha)
}



pooledResidualVariance <- function(M_X, M_Y) {
  # M_X: 3D array of predictors with dimensions (m, n, p)
  # M_Y: Matrix of responses with dimensions (m, n)
  # m = number of imputations, n = number of observations, p = number of predictors

  m <- dim(M_X)[1]
  res_var <- numeric(m)
  beta = matrix(0, nrow = dim(M_X)[1], ncol = dim(M_X)[3])
  alpha = rep(0, dim(M_X)[1])
  for (i in 1:m) {
    # Extract the i-th imputed dataset as a data frame
    df <- as.data.frame(M_X[i, , ])
    colnames(df) <- paste0("x", 1:ncol(df))
    df$y <- M_Y[i, ]

    # Fit a linear model (including an intercept by default)
    fit <- lm(y ~ ., data = df)

    # Extract the residual variance (sigma^2) from the model summary
    res_var[i] <- summary(fit)$sigma^2
    beta[i, ] <- fit$coefficients[-1]
    alpha[i] <- fit$coefficients[1]
  }

  # Pool the residual variance estimates by taking the average
  pooled_res_var <- mean(res_var)

  # Compute an approximate standard error for the pooled residual variance
  pooled_se <- (var(res_var) / m)

  return(list(pooled_res_var = pooled_res_var,
              individual_res_vars = res_var,
              pooled_se = pooled_se, beta = beta, alpha = alpha))
}



#' ARD MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using the Automatic Relevance Determination (ARD) prior
#' across multiply-imputed datasets. The ARD prior imposes feature-specific shrinkage by placing
#' a Gamma prior on the inverse variance (precision) of each coefficient.
#'
#' @param X A 3D array of predictors with dimensions \code{D x n x p}.
#' @param Y A matrix of outcomes with dimensions \code{D x n}.
#' @param standardize Logical; whether to standardize \code{X} within each dataset. Default: \code{FALSE}.
#' @param nburn Number of burn-in iterations. Default: \code{4000}.
#' @param npost Number of posterior samples to store. Default: \code{4000}.
#' @param seed Optional random seed.
#' @param verbose Logical; whether to print MCMC progress. Default: \code{TRUE}.
#' @param printevery Print progress every \code{printevery} iterations. Default: \code{1000}.
#' @param chain_index Integer; index of the current MCMC chain (used for printing). Default: \code{1}.
#'
#' @return A list containing posterior draws of:
#' \describe{
#'   \item{post_beta}{\code{npost x D x p} array of regression coefficients.}
#'   \item{post_alpha}{\code{npost x D} matrix of intercepts.}
#'   \item{post_sigma2}{\code{npost} vector of residual variances.}
#'   \item{post_psi2}{\code{npost x p} matrix of precision parameters for each coefficient.}
#'   \item{post_fitted_Y}{\code{npost x D x n} array of posterior predictive means (with noise).}
#'   \item{post_pool_beta}{\code{(npost * D) x p} matrix; reshaped \code{post_beta} for pooling.}
#'   \item{post_pool_fitted_Y}{\code{(npost * D) x n} matrix; reshaped \code{post_fitted_Y} for pooling.}
#' }
#'
#' @export
#'
#' @importFrom stats as.formula coef dgamma dlnorm gaussian glm lm median plogis predict quantile rbeta rbinom rcauchy residuals rgamma rlnorm rnorm runif sd var vcov


ARD_mcmc = function(X, Y, standardize = F, nburn = 4000, npost = 4000, seed = NULL, verbose = T, printevery = 1000, chain_index = 1){
  if(!is.null(seed))
    set.seed(seed)
  if(standardize){
    X = aperm(simplify2array(lapply(1:dim(X)[1], function(i) scale(X[i, , ]))), c(3, 1, 2))
  }


  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  pool = pooledResidualVariance(X, Y)
  sigma2 = pool$pooled_res_var
  beta = pool$beta
  alpha = pool$alpha

  psi2 = ARD_update_psi2(beta, D, sigma2)

  post_psi2 = matrix(NA, npost, p)
  post_alpha = matrix(NA, npost, D)
  post_sigma2 = rep(NA, npost)
  post_beta = array(NA, dim = c(npost, D, p))
  post_fitted_Y = array(NA, dim = c(npost, D, n))


  for (i in 1:(nburn + npost)) {
    if(i %% printevery == 0 & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", ", ifelse(i <= nburn, "burn-in", "sampling"), sep = ""), "\n")
    if(i == (nburn + 1) & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", sampling", sep = ""), "\n")

    beta = ARD_update_beta(X, Y, alpha, psi2, sigma2)
    alpha = ARD_update_alpha(X, Y, beta, sigma2)
    sigma2 = ARD_update_sigma2(X, Y, beta, alpha, psi2)
    psi2 = ARD_update_psi2(beta, D, sigma2)
    if(i >= nburn){
      post_psi2[i - nburn, ] = psi2
      post_alpha[i - nburn, ] = alpha
      post_sigma2[i - nburn] = sigma2
      post_beta[i - nburn, ,] = beta

      for (d in 1:D) {
        post_fitted_Y[i - nburn, d, ] = X[d, ,] %*% beta[d, ] + alpha[d]+ rnorm(1,0,sqrt(sigma2))
      }
    }
  }
  return(list(
    post_psi2 = post_psi2,
    post_alpha = post_alpha,
    post_sigma2 = post_sigma2,
    post_beta = post_beta,
    post_fitted_Y = post_fitted_Y,
    post_pool_beta = matrix(post_beta, nrow = dim(post_beta)[1] * dim(post_beta)[2], ncol = dim(post_beta)[3]),
    post_pool_fitted_Y = matrix(post_fitted_Y, nrow = dim(post_fitted_Y)[1] * dim(post_fitted_Y)[2], ncol = dim(post_fitted_Y)[3])
  ))
}
