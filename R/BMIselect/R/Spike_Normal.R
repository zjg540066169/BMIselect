
SN_update_gamma = function(X, Y, gamma, alpha, p0, sigma2, v02){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  prob = rep(1, p)
  for(j in 1:p){
    prob[j] = SN_collapse_probability(X, Y, alpha, gamma, sigma2, v02, p0, j)
    gamma[j] = rbinom(1, 1, prob[j])
  }
  return(list(gamma, prob))
}

SN_collapse_probability = function(X, Y, alpha, gamma, sigma2, v02, p0, j){
  D = dim(X)[1]
  # llr = (sum(sapply(1:D, function(d){
  #   X_d = as.matrix(X[d,,,drop = TRUE])
  #   gamma_j = gamma
  #   gamma_j[j] = 1
  #   X_s_d_1 = X_d[,gamma_j == 1,drop = FALSE]
  #   Sigma_1 = diag(1, n) + v02 * X_s_d_1 %*% t(X_s_d_1)
  #
  #   gamma_j[j] = 0
  #   X_s_d_0 = X_d[,gamma_j == 1,drop = FALSE]
  #   Sigma_0 = diag(1, n) + v02 * X_s_d_0 %*% t(X_s_d_0)
  #   0.5 * (determinant(Sigma_0)$modulus[1] - determinant(Sigma_1)$modulus[1] + t(Y[d,] - alpha[d]) %*% (Rfast::spdinv(Sigma_0) - Rfast::spdinv(Sigma_1)) %*% (Y[d,] - alpha[d]) / sigma2)
  # })))


  llr <- sum(sapply(1:D, function(d) {
    Xd    <- X[d,,,drop=TRUE]        # n×p
    resid <- Y[d,] - alpha[d]        # n‐vector
    rj    <- Xd[, j]                 # n‐vector

    # build X0 = Xd[ , gamma_j==1 ] with gamma_j[j]=0
    gamma0 <- gamma; gamma0[j] <- 0
    X0     <- Xd[, gamma0==1, drop=FALSE]

    SN_woodbury_llr_term(X0, rj, resid, v02, sigma2)
  }))


  plogis(log(p0) - log1p(-p0) + llr)
}


SN_woodbury_llr_term <- function(X0, rj, resid, v02, sigma2) {
  # X0     : n×k0 matrix of covariates excluding j
  # rj     : length-n vector, the j-th column of X
  # resid  : length-n vector, Y - alpha
  # v02    : v0^2
  # sigma2 : σ²

  # Special case: no other covariates
  if (ncol(X0) == 0) {
    dj <- 1 + v02 * sum(rj^2)
    mj <- (v02/sigma2) * (sum(rj * resid)^2) / dj
    return(0.5 * (-log(dj) + mj))
  }

  # 1) Form the small ridge matrix and its Cholesky
  M  <- crossprod(X0) + diag(1/v02, ncol(X0))  # = X0^T X0 + v0^{-2}I
  R  <- chol(M)                                 # R^T R = M

  # 2) Compute C rj = rj - X0 %*% solve(M, X0^T rj)
  q_rj <- crossprod(X0, rj)
  z_rj <- backsolve(R, backsolve(R, q_rj, transpose=TRUE))
  cRj  <- rj - (X0 %*% z_rj)                    # <— NO v02 here

  # 3) Compute C resid = resid - X0 %*% solve(M, X0^T resid)
  q_y  <- crossprod(X0, resid)
  z_y  <- backsolve(R, backsolve(R, q_y, transpose=TRUE))
  cCY  <- resid - (X0 %*% z_y)                  # <— NO v02 here

  # 4) Build the two scalars
  dj <- 1 + v02 * sum(rj * cRj)
  mj <- (v02/sigma2) * (sum(rj * cCY)^2) / dj

  # 5) Return half of [–log d_j + m_j]
  0.5 * (-log(dj) + mj)
}







SN_update_sigma2 = function(X, Y, alpha, gamma, v02){
  # X: D x n x p array, Y: D x n, beta: D x p, alpha: vector of length D.
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  a = D * n / 2
  SSE = 0
  if(sum(gamma) == 0){
    for(d in 1:D){
      SSE = SSE + t(Y[d,] - alpha[d]) %*% (Y[d,] - alpha[d])
    }
  }else{
    X_s = X[,,which(gamma == 1),drop = FALSE]
    for(d in 1:D){
      X_s_d = X_s[d,,,drop = TRUE]
      SSE = SSE + t(Y[d,] - alpha[d]) %*% Rfast::spdinv(diag(1, n) + v02 * X_s_d %*% t(X_s_d)) %*% (Y[d,] - alpha[d])
    }
  }
  b = SSE / 2
  return(MCMCpack::rinvgamma(1, a, b))
}


SN_update_beta = function(X, Y, alpha, gamma, v02, sigma2){
  D = dim(X)[1]  # number of imputed datasets
  n = dim(X)[2]  # sample size per dataset
  p = dim(X)[3]  # number of predictors
  beta = matrix(0, nrow = D, ncol = p)  # initialize beta; inactive coefficients will remain zero
  if(sum(gamma) > 0){
    inv_variance = diag(rep(1 / v02, sum(gamma)))
    for(d in 1:D){
      va = Rfast::spdinv((t(as.matrix(X[d,,gamma == 1])) %*% as.matrix(X[d,,gamma == 1]) + inv_variance))
      mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X[d,,gamma == 1])) %*% va
      va = va * sigma2
      beta[d,gamma == 1] = MASS::mvrnorm(1, mu, va)
    }
  }
  return(beta)
}

SN_update_alpha = function(X, Y, gamma, sigma2, v02){
  D = dim(X)[1]
  n = dim(X)[2]
  vector_1 = rep(1, n)
  alpha = rep(0, D)

  if(sum(gamma) == 0){
    for(d in 1:D){
      mu_alpha = mean(Y[d,])
      var_alpha = sigma2 / n
      alpha[d] = rnorm(1, mean = mu_alpha, sd = sqrt(var_alpha))
    }
  }else{
    X_s = X[,,which(gamma == 1),drop = FALSE]
    for(d in 1:D){
      X_s_d = X_s[d,,,drop = TRUE]
      mu_alpha = t(vector_1) %*% Rfast::spdinv(diag(1, n) + v02 * X_s_d %*% t(X_s_d)) %*% Y[d,] / (t(vector_1) %*% Rfast::spdinv(diag(1, n) + v02 * X_s_d %*% t(X_s_d)) %*% vector_1)
      var_alpha = sigma2 / (t(vector_1) %*% Rfast::spdinv(diag(1, n) + v02 * X_s_d %*% t(X_s_d)) %*% vector_1)
      alpha[d] = rnorm(1, mean = mu_alpha, sd = sqrt(var_alpha))
    }
  }
  return(alpha)
}


# collapse_likelihood = function(X, Y, alpha, gamma, sigma2, v02){
#   log_LL = 0
#   D = dim(X)[1]
#   n = dim(X)[2]
#   if(sum(gamma) == 0){
#     for (d in 1:D) {
#       log_LL = log_LL + mvnfast::dmvn(Y[d,], rep(alpha[d], n), sigma2 * diag(1, n), log = TRUE)
#     }
#   }else{
#     X_s = X[, , which(gamma == 1), drop = FALSE]
#
#     for (d in 1:D) {
#       X_s_d = as.matrix(X_s[d,,,drop = TRUE])
#       log_LL = log_LL + mvnfast::dmvn(Y[d,], rep(alpha[d], n), sigma2 * (diag(1, n) + v02 * X_s_d %*% t(X_s_d)), log = TRUE)
#     }
#   }
#   return(log_LL)
# }


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



#' Spike-and-Normal MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using a spike-and-normal prior:
#' each coefficient is either zero (with probability \code{1 - p0}) or drawn
#' from a normal distribution. This is performed in a collapsed Gibbs sampler across multiply-imputed datasets.
#'
#' @param X A 3D array of predictors with dimensions \code{D x n x p}.
#' @param Y A matrix of outcomes with dimensions \code{D x n}.
#' @param standardize Logical; whether to standardize \code{X} within each dataset. Default: \code{TRUE}.
#' @param p0 Prior inclusion probability for each coefficient. Default: \code{0.5}.
#' @param v02 Prior slab variance multiplier. Default: \code{2}.
#' @param nburn Number of burn-in iterations. Default: \code{4000}.
#' @param npost Number of posterior samples to store. Default: \code{4000}.
#' @param seed Optional random seed.
#' @param verbose Logical; whether to print MCMC progress. Default: \code{TRUE}.
#' @param printevery Print progress every \code{printevery} iterations. Default: \code{1000}.
#' @param chain_index Integer; index of the current MCMC chain (used for printing). Default: \code{1}.
#'
#' @return A list containing posterior draws of:
#' \describe{
#'   \item{post_logit_inclusion_prob}{\code{npost x p} matrix of logit-transformed inclusion probabilities.}
#'   \item{post_gamma}{\code{npost x p} binary inclusion indicators.}
#'   \item{post_alpha}{\code{npost x D} matrix of dataset-specific intercepts.}
#'   \item{post_sigma2}{\code{npost} vector of residual variances.}
#'   \item{post_beta}{\code{npost x D x p} array of regression coefficients (zero for excluded variables).}
#'   \item{post_fitted_Y}{\code{npost x D x n} array of posterior predictive means (with noise).}
#'   \item{post_pool_beta}{\code{(npost * D) x p} matrix; reshaped \code{post_beta} for pooling.}
#'   \item{post_pool_fitted_Y}{\code{(npost * D) x n} matrix; reshaped \code{post_fitted_Y} for pooling.}
#'   \item{p0}{Prior inclusion probability used.}
#'   \item{v02}{Slab variance multiplier used.}
#' }
#'
#' @export
#'
#' @importFrom stats as.formula coef dgamma dlnorm gaussian glm lm median plogis predict quantile rbeta rbinom rcauchy residuals rgamma rlnorm rnorm runif sd var vcov

spike_normal_mcmc = function(X, Y, standardize = T, p0 = 0.5, v02 = NULL, nburn = 4000, npost = 4000, seed = NULL, verbose = T, printevery = 1000, chain_index = 1){
  if(!is.null(seed))
    set.seed(seed)
  if(standardize){
    X = aperm(simplify2array(lapply(1:dim(X)[1], function(i) scale(X[i, , ]))), c(3, 1, 2))
  }

  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  if(is.null(v02)){
    v02 = 2
  }

  pool = pooledResidualVariance(X, Y)
  sigma2 = pool$pooled_res_var
  alpha = pool$alpha
  gamma = rbinom(p, 1, p0)


  post_gamma = matrix(NA, npost, p)
  post_inclusion_prob = matrix(NA, npost, p)
  post_alpha = matrix(NA, npost, D)
  post_sigma2 = rep(NA, npost)
  post_fitted_Y = array(NA, dim = c(npost, D, n))
  post_beta = array(NA, dim = c(npost, D, p))

  for (i in 1:(nburn + npost)) {
    if(i %% printevery == 0 & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", ", ifelse(i <= nburn, "burn-in", "sampling"), sep = ""), "\n")
    if(i == (nburn + 1) & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", sampling", sep = ""), "\n")
    gamma_list = SN_update_gamma(X, Y, gamma, alpha, p0, sigma2, v02)
    gamma = gamma_list[[1]]
    inclusion_prob = gamma_list[[2]]
    alpha = SN_update_alpha(X, Y, gamma, sigma2, v02)
    sigma2 = SN_update_sigma2(X, Y, alpha, gamma, v02)
    beta = SN_update_beta(X, Y, alpha, gamma, v02, sigma2)

    if(i > nburn){
      idx = i - nburn
      post_gamma[idx, ] = gamma
      post_alpha[idx, ] = alpha
      post_sigma2[idx] = sigma2
      post_inclusion_prob[idx,] = inclusion_prob
      post_beta[i - nburn, ,] = beta

      for (d in 1:D) {
        #post_fitted_Y[idx, d, ] = alpha[d]
        post_fitted_Y[i - nburn, d, ] = X[d, ,] %*% beta[d, ] + alpha[d] + rnorm(1,0,sqrt(sigma2))
      }
    }
  }

  return(list(
    post_logit_inclusion_prob = arm::logit(post_inclusion_prob),
    post_gamma = post_gamma,
    post_alpha = post_alpha,
    post_sigma2 = post_sigma2,
    post_beta = post_beta,
    post_fitted_Y = post_fitted_Y,
    post_pool_beta = matrix(post_beta, nrow = dim(post_beta)[1] * dim(post_beta)[2], ncol = dim(post_beta)[3]),
    post_pool_fitted_Y = matrix(post_fitted_Y, nrow = dim(post_fitted_Y)[1] * dim(post_fitted_Y)[2],
                                ncol = dim(post_fitted_Y)[3]),
    p0 = p0, v02 = v02
  ))
}
