ARD_update_psi2 = function(beta, beta_mul, D, sigma2, eps = 1e-6){
  psi2 = sapply(1:dim(beta)[2], function(j) stats::rgamma(1, D / 2, scale = 2 * sigma2 / (beta_mul[j])))
  psi2 = pmin(1/eps, psi2)
  return(psi2)
}

ARD_update_sigma2 = function(Y, Xbeta, beta_mul, alpha, psi2){
  D = dim(Y)[1]
  n = dim(Y)[2]
  p = length(beta_mul)
  a =  D * (n + p) / 2
  SSE = 0
  SSE_beta = 0
  for(d in 1:D){
    y_hat = Xbeta[d,] + alpha[d]
    SSE = SSE + sum((Y[d,] - y_hat)^2)
  }

  for(j in 1:p){
    SSE_beta = SSE_beta + beta_mul[j] * psi2[j]
  }

  b = (SSE + SSE_beta) / 2
  return(MCMCpack::rinvgamma(1, a, b))
}

ARD_update_beta = function(X, Y, XtX, alpha, psi2, sigma2){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  beta = matrix(0, nrow = D, ncol = p)
  hat_matrix_proj = array(0, dim = c(D, n, n))
  inv_variance = diag(psi2)
  for(d in 1:D){
    va = Rfast::spdinv(XtX[d,,] + inv_variance)
    hat_matrix_proj[d,,] = as.matrix(X[d,,]) %*% va %*% t(as.matrix(X[d,,]))
    mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X[d,,])) %*% va
    va = va * sigma2
    beta[d,] = MASS::mvrnorm(1, mu, va)
  }
  return(list(beta = beta, hat_matrix_proj = hat_matrix_proj))
}


ARD_update_alpha = function(Xbeta, Y, sigma2){
  D = dim(Y)[1]
  n = dim(Y)[2]
  alpha = rep(0, D)
  for(d in 1:D){
    va = sigma2 / n
    mu = mean(Y[d,] - Xbeta[d,])
    alpha[d] = stats::rnorm(1, mu, sqrt(va))
  }
  return(alpha)
}


# Compute pooled residual variance and initialize beta/alpha
pooledResidualVariance <- function(M_X, M_Y, intercept = FALSE) {
  m <- dim(M_X)[1]
  res_var <- numeric(m)
  beta <- matrix(0, m, dim(M_X)[3])
  alpha <- numeric(m)
  for (i in 1:m) {
    df <- as.data.frame(M_X[i,,])
    colnames(df) <- paste0("x", 1:ncol(df))
    df$y <- M_Y[i, ]
    if(intercept == FALSE){
      fit <- lm(y ~ 0 + ., data = df)
      beta[i, ] <- fit$coefficients
      alpha[i] <- 0
    }else{
      fit <- lm(y ~ ., data = df)
      beta[i, ] <- fit$coefficients[-1]
      alpha[i] <- fit$coefficients[1]
    }
    res_var[i] <- summary(fit)$sigma^2
  }
  pooled_res_var <- mean(res_var)
  pooled_se <- sqrt(var(res_var) / m)
  return(list(pooled_res_var = pooled_res_var,
              individual_res_vars = res_var,
              pooled_se = pooled_se,
              beta = beta,
              alpha = alpha))
}



#' ARD MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using the Automatic Relevance Determination (ARD) prior
#' across multiply-imputed datasets. The ARD prior imposes feature-specific shrinkage by placing
#' a prior proportional to inverse of precision of each coefficient.
#'
#' @param X A 3-D array of predictors with dimensions \code{D × n × p}.
#' @param Y A matrix of outcomes with dimensions \code{D × n}.
#' @param intercept Logical; include an intercept? Default \code{TRUE}.
#' @param nburn Integer; number of burn-in MCMC iterations. Default \code{4000}.
#' @param npost Integer; number of post-burn-in samples to retain. Default \code{4000}.
#' @param seed Integer or \code{NULL}; random seed for reproducibility. Default \code{NULL}.
#' @param verbose Logical; print progress messages? Default \code{TRUE}.
#' @param printevery Integer; print progress every this many iterations. Default \code{1000}.
#' @param chain_index Integer; index of this MCMC chain (for labeling messages). Default \code{1}.
#'
#' @return A named \code{list} with components:
#' \describe{
#'   \item{\code{post_beta}}{Array \code{npost × D × p} of sampled regression coefficients.}
#'   \item{\code{post_alpha}}{Matrix \code{npost × D} of sampled intercepts (if used).}
#'   \item{\code{post_sigma2}}{Numeric vector length \code{npost}, sampled residual variances.}
#'   \item{\code{post_psi2}}{Matrix \code{npost × p} of sampled precision parameters for each coefficient.}
#'   \item{\code{post_fitted_Y}}{Array \code{npost × D × n} of posterior predictive draws (with noise).}
#'   \item{\code{post_pool_beta}}{Matrix \code{(npost * D) × p} of pooled coefficient draws.}
#'   \item{\code{post_pool_fitted_Y}}{Matrix \code{(npost * D) × n} of pooled predictive draws (with noise).}
#'   \item{\code{hat_matrix_proj}}{Matrix \code{D × n × n} of averaged projection hat-matrices. To avoid recalculate for estimating degree of freedom.}
#' }
#'
#' @examples
#' sim <- sim_B(n = 100, p = 20, type = "MAR", SNP = 1.5, corr = 0.5,
#' low_missing = TRUE, n_imp = 5, seed = 123)
#' X <- sim$data_MI$X
#' Y <- sim$data_MI$Y
#' fit <- ARD_mcmc(X, Y, nburn = 100, npost = 100)
#' @export
ARD_mcmc = function(X, Y, intercept = TRUE, nburn = 4000, npost = 4000, seed = NULL, verbose = TRUE, printevery = 1000, chain_index = 1){
  if(!is.null(seed))
    set.seed(seed)


  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  if(n > p){
    pool = pooledResidualVariance(X, Y, intercept)
    sigma2 = pool$pooled_res_var
    beta = pool$beta
    alpha = pool$alpha
  }else{
    pool = pooledResidualVariance(array(1, dim = c(D,n,1)), Y, intercept)
    sigma2 = pool$pooled_res_var
    beta = array(0, dim = c(D, p))
    alpha = rep(0, D)
  }

  hat_matrix_proj = array(0, dim = c(D, n, n))
  XtX = array(NA, dim = c(D, p, p))

  for (d in 1:D) {
    XtX[d,,] = t(X[d,,,drop = TRUE]) %*% X[d,,,drop = TRUE]
  }

  Xbeta = matrix(NA, D, n)
  for (d in 1:D) {
    Xbeta[d,] = X[d,,] %*% beta[d, ]
  }

  beta_mul = sapply(1:p, function(j) t(beta[,j]) %*% beta[,j])


  psi2 = ARD_update_psi2(beta, beta_mul, D, sigma2)

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

    if(intercept)
      alpha = ARD_update_alpha(Xbeta, Y, sigma2)

    sigma2 = ARD_update_sigma2(Y, Xbeta, beta_mul, alpha, psi2)
    psi2 = ARD_update_psi2(beta, beta_mul, D, sigma2)

    beta_list = ARD_update_beta(X, Y, XtX, alpha, psi2, sigma2)
    beta = beta_list$beta
    for (d in 1:D) {
      Xbeta[d,] = X[d,,] %*% beta[d, ]
    }
    beta_mul = sapply(1:p, function(j) t(beta[,j]) %*% beta[,j])

    if(i >= nburn){
      hat_matrix_proj = hat_matrix_proj + beta_list$hat_matrix_proj
      post_psi2[i - nburn, ] = psi2
      post_alpha[i - nburn, ] = alpha
      post_sigma2[i - nburn] = sigma2
      post_beta[i - nburn, ,] = beta

      for (d in 1:D) {
        post_fitted_Y[i - nburn, d, ] = Xbeta[d,] + alpha[d]+ stats::rnorm(n,0,sqrt(sigma2))
      }
    }
  }
  hat_matrix_proj = hat_matrix_proj / npost
  return(list(
    post_psi2 = post_psi2,
    post_alpha = post_alpha,
    post_sigma2 = post_sigma2,
    post_beta = post_beta,
    post_fitted_Y = post_fitted_Y,
    post_pool_beta = matrix(post_beta, nrow = dim(post_beta)[1] * dim(post_beta)[2], ncol = dim(post_beta)[3]),
    post_pool_fitted_Y = matrix(post_fitted_Y, nrow = dim(post_fitted_Y)[1] * dim(post_fitted_Y)[2], ncol = dim(post_fitted_Y)[3]),
    hat_matrix_proj = hat_matrix_proj
  ))
}
