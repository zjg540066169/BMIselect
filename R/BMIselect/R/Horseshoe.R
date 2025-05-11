H_update_eta = function(tau2){
  eta = MCMCpack::rinvgamma(1, 1, 1 + 1 / tau2)
  return(eta)
}

H_update_tau2 = function(beta, lambda2, sigma2, eta){
  # beta: D x p matrix (each row is one imputation)
  D = dim(beta)[1]
  p = dim(beta)[2]
  a = (D * p + 1) / 2
  b = 1 / eta + 1 / (2 * sigma2) * sum(sapply(1:p, function(j) t(beta[,j]) %*% beta[,j] / lambda2[j]))
  return(MCMCpack::rinvgamma(1, a, b))
}

H_update_kappa = function(lambda2){
  kappa = rep(0, length(lambda2))
  for (i in 1:length(lambda2)) {
    kappa[i] = MCMCpack::rinvgamma(1, 1, 1 + 1 / lambda2[i])
  }
  return(kappa)
}

H_update_lambda2 = function(beta, D, kappa, tau2, sigma2){
  p = dim(beta)[2]
  lambda2 = rep(0, p)
  for (i in 1:p) {
    lambda2[i] = MCMCpack::rinvgamma(1, (D+1) / 2, 1/kappa[i] + t(beta[,i]) %*% beta[,i] / (2 * sigma2 * tau2))
  }
  return(lambda2)
}

H_update_sigma2 = function(X, Y, beta, alpha, lambda2, tau2){
  # X: D x n x p array, Y: D x n, beta: D x p, alpha: vector of length D.
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  a = D * (n + p) / 2
  SSE = 0
  SSE_beta = 0

  for(d in 1:D){
    y_hat = as.matrix(X[d,,]) %*% beta[d,] + alpha[d]
    SSE = SSE + sum((Y[d,] - y_hat)^2)
  }

  for(j in 1:p){
    SSE_beta = SSE_beta + t(beta[,j]) %*% beta[,j] / (lambda2[j] * tau2)
  }

  b = (SSE + SSE_beta) / 2
  return(MCMCpack::rinvgamma(1, a, b))
}

H_update_beta = function(X, Y, alpha, lambda2, tau2, sigma2){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  beta = matrix(0, nrow = D, ncol = p)
  inv_variance = diag(1 / (lambda2 * tau2))
  for(d in 1:D){
    va = Rfast::spdinv((t(X[d,,]) %*% as.matrix(X[d,,]) + inv_variance))
    mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X[d,,])) %*% va
    va = va * sigma2
    beta[d,] = MASS::mvrnorm(1, mu, va)
  }
  return(beta)
}

H_update_alpha = function(X, Y, beta, sigma2){
  D = dim(X)[1]
  n = dim(X)[2]
  alpha = rep(0, D)
  for(d in 1:D){
    var_alpha = sigma2 / n
    mu_alpha = mean(Y[d, ] - as.matrix(X[d, , ]) %*% beta[d, ])
    alpha[d] = rnorm(1, mean = mu_alpha, sd = sqrt(var_alpha))
  }
  return(alpha)
}

pooledResidualVariance <- function(M_X, M_Y) {
  # M_X: 3D array of predictors (m x n x p), M_Y: matrix (m x n)
  m <- dim(M_X)[1]
  res_var <- numeric(m)
  beta = matrix(0, nrow = m, ncol = dim(M_X)[3])
  alpha = rep(0, m)
  for (i in 1:m) {
    df <- as.data.frame(M_X[i, , ])
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
              pooled_se = pooled_se, beta = beta, alpha = alpha))
}



#' Horseshoe MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using the hierarchical Horseshoe prior
#' across multiply-imputed datasets. This model applies global-local shrinkage
#' to regression coefficients using a global parameter \eqn{\tau^2}, local parameters
#' \eqn{\lambda_j^2}, and auxiliary hyperpriors on both.
#'
#' @param X A 3D array of predictors with dimensions \code{D x n x p}.
#' @param Y A matrix of outcomes with dimensions \code{D x n}.
#' @param standardize Logical; whether to standardize \code{X} within each dataset. Default: \code{TRUE}.
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
#'   \item{post_lambda2}{\code{npost x p} matrix of local shrinkage parameters \eqn{\lambda_j^2}.}
#'   \item{post_kappa}{\code{npost x p} matrix of auxiliary parameters \eqn{\kappa_j}.}
#'   \item{post_tau2}{\code{npost} vector of global scale parameters \eqn{\tau^2}.}
#'   \item{post_eta}{\code{npost} vector of auxiliary global shrinkage variables \eqn{\eta}.}
#'   \item{post_fitted_Y}{\code{npost x D x n} array of posterior predictive means (with noise).}
#'   \item{post_pool_beta}{\code{(npost * D) x p} matrix; reshaped \code{post_beta} for pooling.}
#'   \item{post_pool_fitted_Y}{\code{(npost * D) x n} matrix; reshaped \code{post_fitted_Y} for pooling.}
#' }
#'
#' @export
#'
#' @importFrom stats as.formula coef dgamma dlnorm gaussian glm lm median plogis predict quantile rbeta rbinom rcauchy residuals rgamma rlnorm rnorm runif sd var vcov

horseshoe_mcmc = function(X, Y, standardize = T, nburn = 4000, npost = 4000, seed = NULL, verbose = T, printevery = 1000, chain_index = 1){
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

  # Initialize global and local scales using 95th quantile of Cauchy squared.
  tau2 = rcauchy(1)^2
  lambda2 = rcauchy(p)^2

  eta = H_update_eta(tau2)
  kappa = H_update_kappa(lambda2)

  post_lambda2 = matrix(NA, npost, p)
  post_kappa = matrix(NA, npost, p)
  post_tau2 = rep(NA, npost)
  post_eta = rep(NA, npost)
  post_alpha = matrix(NA, npost, D)
  post_sigma2 = rep(NA, npost)
  post_beta = array(NA, dim = c(npost, D, p))
  post_fitted_Y = array(NA, dim = c(npost, D, n))

  for (i in 1:(nburn + npost)) {
    if(i %% printevery == 0 & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", ", ifelse(i <= nburn, "burn-in", "sampling"), sep = ""), "\n")
    if(i == (nburn + 1) & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", sampling", sep = ""), "\n")

    eta = H_update_eta(tau2)
    tau2 = H_update_tau2(beta, lambda2, sigma2, eta)
    kappa = H_update_kappa(lambda2)
    lambda2 = H_update_lambda2(beta, D, kappa, tau2, sigma2)

    beta = H_update_beta(X, Y, alpha, lambda2, tau2, sigma2)
    alpha = H_update_alpha(X, Y, beta, sigma2)
    sigma2 = H_update_sigma2(X, Y, beta, alpha, lambda2, tau2)

    if(i > nburn){
      idx = i - nburn
      post_lambda2[idx, ] = lambda2
      post_kappa[idx, ] = kappa
      post_tau2[idx] = tau2
      post_eta[idx] = eta
      post_alpha[idx, ] = alpha
      post_sigma2[idx] = sigma2
      post_beta[idx, , ] = beta

      for (d in 1:D) {
        post_fitted_Y[idx, d, ] = X[d, , ] %*% beta[d, ] + alpha[d]+ rnorm(1,0,sqrt(sigma2))
      }
    }
  }

  return(list(
    post_lambda2 = post_lambda2,
    post_kappa = post_kappa,
    post_tau2 = post_tau2,
    post_eta = post_eta,
    post_alpha = post_alpha,
    post_sigma2 = post_sigma2,
    post_beta = post_beta,
    post_fitted_Y = post_fitted_Y,
    post_pool_beta = matrix(post_beta, nrow = dim(post_beta)[1] * dim(post_beta)[2],
                            ncol = dim(post_beta)[3]),
    post_pool_fitted_Y = matrix(post_fitted_Y, nrow = dim(post_fitted_Y)[1] * dim(post_fitted_Y)[2],
                                ncol = dim(post_fitted_Y)[3])
  ))
}
