# Update global shrinkage parameter rho
ML_update_rho = function(lambda2, D, p, h, s) {
  a = h + p * (D + 1) / 2
  t = 1 / (1 / s + D * sum(lambda2) / 2)
  return(rgamma(1, a, scale = t))
}

# Update local shrinkage lambda2 (per predictor)
ML_update_lambda2 = function(beta, beta_mul, D, rho, sigma2) {
  lambda2 = sapply(1:dim(beta)[2], function(j) GIGrvg::rgig(1, 0.5, beta_mul[j] / sigma2, D * rho))
  return(lambda2)
}

# Update residual variance sigma2
ML_update_sigma2 = function(X, Y, Xbeta, beta_mul, alpha, lambda2) {
  D = dim(X)[1]; n = dim(X)[2]; p = dim(X)[3]
  a = D * (n + p) / 2
  SSE = sum(sapply(1:D, function(d) sum((Y[d, ] - Xbeta[d,] - alpha[d])^2)))
  SSE_beta = sum(sapply(1:p, function(j) beta_mul[j] / lambda2[j]))
  b = (SSE + SSE_beta) / 2
  return(MCMCpack::rinvgamma(1, a, b))
}

# Update regression coefficients beta
ML_update_beta = function(X, Y, XtX, alpha, lambda2, sigma2) {
  D = dim(X)[1]; n = dim(X)[2]; p = dim(X)[3]
  beta = matrix(0, D, p)
  hat_matrix_proj = array(0, dim = c(D, n, n))
  inv_variance = diag(1 / lambda2)
  for (d in 1:D) {
    va = Rfast::spdinv(XtX[d,,] + inv_variance)
    hat_matrix_proj[d,,] = as.matrix(X[d,,]) %*% va %*% t(as.matrix(X[d,,]))
    mu = (t(Y[d,] - alpha[d]) %*% X[d,,]) %*% va
    beta[d,] = MASS::mvrnorm(1, mu, va * sigma2)
  }
  return(list(beta = beta, hat_matrix_proj = hat_matrix_proj))
}

# Update intercepts alpha
ML_update_alpha = function(X, Y, Xbeta, sigma2) {
  D = dim(X)[1]; n = dim(X)[2]
  alpha = numeric(D)
  for (d in 1:D) {
    mu = mean(Y[d, ] - Xbeta[d,])
    alpha[d] = rnorm(1, mu, sqrt(sigma2 / n))
  }
  return(alpha)
}

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


#' Multi-Laplace MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection under the Multi-Laplace prior on
#' regression coefficients across multiply-imputed datasets.  The prior shares
#' local shrinkage parameters (\code{lambda2}) across imputations and places
#' a Gamma(\code{h}, \code{v}) hyperprior on the global parameter \code{rho}.
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
#'   \item{\code{hat_matrix_proj}}{Matrix \code{D × n × n} of averaged projection hat-matrices. To avoid recalculate for estimating degree of freedom.}
#'   \item{\code{h}, \code{v}}{Numeric; the shape and scale hyperparameters used.}
#' }
#' @examples
#' sim <- sim_B(n = 100, p = 20, type = "MAR", SNP = 1.5, corr = 0.5, low_missing = TRUE,
#' n_imp = 5, seed = 123)
#' X <- sim$data_MI$X
#' Y <- sim$data_MI$Y
#' fit <- multi_laplace_mcmc(X, Y, intercept = TRUE, nburn = 100, npost = 100)
#' @export
multi_laplace_mcmc = function(X, Y, intercept = TRUE, h = 2, v = NULL, nburn = 4000, npost = 4000, seed = NULL, verbose = TRUE, printevery = 1000, chain_index = 1) {
  if (!is.null(seed)) set.seed(seed)

  D = dim(X)[1]; n = dim(X)[2]; p = dim(X)[3]
  if (is.null(v)) v = (D + 1) / D / (h - 1)

  rho = rgamma(1, shape = h, scale = v)
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

  beta_mul = sapply(1:p, function(j) t(beta[,j]) %*% beta[,j])
  lambda2 = ML_update_lambda2(beta, beta_mul, D, rho, sigma2)

  post_lambda2 = matrix(NA, npost, p)
  post_alpha = matrix(NA, npost, D)
  post_rho = rep(NA, npost)
  post_sigma2 = rep(NA, npost)
  post_beta = array(NA, dim = c(npost, D, p))
  post_fitted_Y = array(NA, dim = c(npost, D, n))

  hat_matrix_proj = array(0, dim = c(D, n, n))
  XtX = array(NA, dim = c(D, p, p))

  for (d in 1:D) {
    XtX[d,,] = t(X[d,,,drop = TRUE]) %*% X[d,,,drop = TRUE]
  }

  Xbeta = matrix(NA, D, n)
  for (d in 1:D) {
    Xbeta[d,] = X[d,,] %*% beta[d, ]
  }



  for (i in 1:(nburn + npost)) {
    if (i %% printevery == 0 && verbose)
      cat(sprintf("Chain %d: %d / %d (%s)\n", chain_index, i, nburn + npost, ifelse(i <= nburn, "burn-in", "sampling")))
    if (i == (nburn + 1) && verbose)
      cat(sprintf("Chain %d: %d / %d (sampling)\n", chain_index, i, nburn + npost))

    rho = ML_update_rho(lambda2, D, p, h, v)
    lambda2 = ML_update_lambda2(beta, beta_mul, D, rho, sigma2)
    if(intercept)
      alpha = ML_update_alpha(X, Y, Xbeta, sigma2)
    sigma2 = ML_update_sigma2(X, Y, Xbeta, beta_mul, alpha, lambda2)

    beta_list = ML_update_beta(X, Y, XtX, alpha, lambda2, sigma2)
    beta = beta_list$beta
    for (d in 1:D) {
      Xbeta[d,] = X[d,,] %*% beta[d, ]
    }
    beta_mul = sapply(1:p, function(j) t(beta[,j]) %*% beta[,j])

    if (i >= nburn) {
      hat_matrix_proj = hat_matrix_proj + beta_list$hat_matrix_proj
      post_lambda2[(i - nburn), ] = lambda2
      post_alpha[(i - nburn), ] = alpha
      post_rho[(i - nburn)] = rho
      post_sigma2[(i - nburn)] = sigma2
      post_beta[(i - nburn), ,] = beta

      for (d in 1:D) {
        post_fitted_Y[(i - nburn), d, ] = Xbeta[d,] + alpha[d] + stats::rnorm(n, 0, sqrt(sigma2))
      }
    }
  }
  hat_matrix_proj = hat_matrix_proj / npost
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
    v = v,
    hat_matrix_proj = hat_matrix_proj
  ))
}
