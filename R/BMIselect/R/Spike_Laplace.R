# SL_update_p0 = function(pi, a = 1, b = 1){
#   return(rbeta(1, a + sum(pi), b + sum(1 - pi)))
# }
#
# SL_update_pi = function(beta, p0, sigma2, tau2){
#   D = dim(beta)[1]
#   p = dim(beta)[2]
#   pi = rep(0, p)
#   for (j in 1:p) {
#     Sigma = diag(sigma2 * tau2[j], D)
#     success = p0 * mixtools::dmvnorm(beta[,j], 0, Sigma)
#     fail = (1 - p0) * prod(beta[,j] == 0)
#     pi[j] = rbinom(1, 1, success / (success + fail))
#   }
#   return(pi)
# }
#
# SL_update_tau2 = function(beta, lambda, sigma2){
#   D = dim(beta)[1]
#   p = dim(beta)[2]
#   tau2 = rep(0, p)
#   for (j in 1:p) {
#     tau2[j] = GIGrvg::rgig(1, 0.5, t(beta[,j]) %*% beta[,j] / sigma2, lambda)
#   }
#   return(tau2)
# }
#
# SL_update_beta = function(X, Y, alpha, pi, tau2, sigma2, eps = 1e-6){
#   D = dim(X)[1]  # number of imputed datasets
#   n = dim(X)[2]  # sample size per dataset
#   p = dim(X)[3]  # number of predictors
#   beta = matrix(0, nrow = D, ncol = p)  # initialize beta; inactive coefficients will remain zero
#
#   for(d in 1:D){
#     # Extract design matrix and response for imputation d:
#     X_d = as.matrix(X[d, , ])
#     active = which(pi == 1)
#     if(length(active) > 0){
#       va = Rfast::spdinv((t(X_d[,active, drop = FALSE]) %*% as.matrix(X_d[ ,active, drop = FALSE]) +  diag(1 / tau2[active]))) * sigma2
#       mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X_d[,active, drop = FALSE])) %*% va / sigma2
#       beta[d, active] = MASS::mvrnorm(1, mu, va)
#       #
#       #
#       # # For active predictors, form submatrix of predictors:
#       # X_active = X_d[, active, drop = FALSE]
#       # V_active = Rfast::spdinv(t(X_active) %*% X_active + diag(1 / tau2[active]))    #+ diag(eps, length(active)) )
#       # Sigma_active = sigma2 * V_active
#       # m_active = V_active %*% t(X_active) %*% (y_d - alpha[d])
#       #
#       # # Draw from the multivariate normal:
#       # beta_active = as.vector(MASS::mvrnorm(1, mu = m_active, Sigma = Sigma_active))
#       #
#       # # Set active entries in beta; inactive ones remain 0 (spike)
#       # beta[d, active] = beta_active
#     }
#   }
#   return(beta)
# }
#
# SL_update_alpha = function(X, Y, beta, sigma2){
#   D = dim(X)[1]
#   n = dim(X)[2]
#   alpha = rep(0, D)
#   for(d in 1:D){
#     var_alpha = sigma2 / n
#     mu_alpha = mean(Y[d, ] - as.matrix(X[d, , ]) %*% beta[d, ])
#     alpha[d] = rnorm(1, mean = mu_alpha, sd = sqrt(var_alpha))
#   }
#   return(alpha)
# }
#
# SL_update_sigma2 = function(X, Y, beta, alpha, pi, tau2){
#   # X: D x n x p array, Y: D x n, beta: D x p, alpha: vector of length D.
#   D = dim(X)[1]
#   n = dim(X)[2]
#   p = dim(X)[3]
#   a = D * (n + p) / 2
#   SSE = 0
#   SSE_beta = 0
#   # The prior for β uses variance σ² * τ² * λ_j², so its precision is diag(1/(tau2 * lambdas2))
#   for(d in 1:D){
#     y_hat = as.matrix(X[d, , ]) %*% beta[d, ] + alpha[d]
#     SSE = SSE + sum((Y[d, ] - y_hat)^2)
#     SSE_beta = SSE_beta + t(beta[d, ]) %*% diag(pi / tau2) %*% beta[d, ]
#   }
#
#   b = (SSE + SSE_beta) / 2
#   return(MCMCpack::rinvgamma(1, a, b))
# }

SL_update_alpha = function(X, Y, Z, sigma2, lambda2){
  D = dim(X)[1]
  n = dim(X)[2]
  vector_1 = rep(1, n)
  alpha = rep(0, D)

  if(sum(Z) == 0){
    for(d in 1:D){
      mu_alpha = mean(Y[d,])
      var_alpha = sigma2 / n
      alpha[d] = rnorm(1, mean = mu_alpha, sd = sqrt(var_alpha))
    }
  }else{
    X_s = X[,,which(Z == 1),drop = FALSE]
    Lambda_Z = diag(lambda2[Z==1])
    for(d in 1:D){
      X_s_d = X_s[d,,,drop = TRUE]
      mu_alpha = t(vector_1) %*% Rfast::spdinv(diag(1, n) + X_s_d %*% Lambda_Z %*% t(X_s_d)) %*% Y[d,] / (t(vector_1) %*% Rfast::spdinv(diag(1, n) + X_s_d %*% Lambda_Z %*% t(X_s_d)) %*% vector_1)
      var_alpha = sigma2 / (t(vector_1) %*% Rfast::spdinv(diag(1, n) + X_s_d %*% Lambda_Z %*% t(X_s_d)) %*% vector_1)
      alpha[d] = rnorm(1, mean = mu_alpha, sd = sqrt(var_alpha))
    }
  }
  return(alpha)
}



SL_update_beta = function(X, Y, alpha, Z, lambda2, sigma2){
  D = dim(X)[1]  # number of imputed datasets
  n = dim(X)[2]  # sample size per dataset
  p = dim(X)[3]  # number of predictors
  beta = matrix(0, nrow = D, ncol = p)  # initialize beta; inactive coefficients will remain zero
  if(sum(Z) > 0){
    inv_variance = diag(1/lambda2[Z == 1])
    for(d in 1:D){
      va = Rfast::spdinv((t(as.matrix(X[d,,Z == 1])) %*% as.matrix(X[d,,Z == 1]) + inv_variance))
      mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X[d,,Z == 1])) %*% va
      va = va * sigma2
      beta[d,Z == 1] = MASS::mvrnorm(1, mu, va)
    }
  }
  return(beta)
}

SL_update_sigma2 = function(X, Y, alpha, Z, lambda2){
  # X: D x n x p array, Y: D x n, beta: D x p, alpha: vector of length D.
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  a = D * n / 2
  SSE = 0
  Lambda_Z = diag(lambda2[Z == 1])
  if(sum(Z) == 0){
    for(d in 1:D){
      SSE = SSE + t(Y[d,] - alpha[d]) %*% (Y[d,] - alpha[d])
    }
  }else{
    X_s = X[,,which(Z == 1),drop = FALSE]
    for(d in 1:D){
      X_s_d = X_s[d,,,drop = TRUE]
      SSE = SSE + t(Y[d,] - alpha[d]) %*% Rfast::spdinv(diag(1, n) + X_s_d %*% Lambda_Z %*% t(X_s_d)) %*% (Y[d,] - alpha[d])
    }
  }
  b = SSE / 2
  return(MCMCpack::rinvgamma(1, a, b))
}


SL_update_rho = function(lambda2, D, p, a, b){
  shape = a + p * (D + 1) / 2
  scale = 1 / (1 / b + D * sum(lambda2) / 2)
  return(rgamma(1, shape, scale = scale))
}

SL_update_Z = function(X, Y, alpha, Z, sigma2, lambda2, theta){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  prob = rep(1, p)
  for (j in 1:p) {
    Z[j] = 1
    R_j = SL_collapse_likelihood(X, Y, alpha, Z, sigma2, lambda2, log = TRUE)
    Z[j] = 0
    R_j = R_j - SL_collapse_likelihood(X, Y, alpha, Z, sigma2, lambda2, log = TRUE)
    prob[j] = theta[j] / (theta[j] + (1 - theta[j]) * exp(-R_j))
    Z[j] = rbinom(1, 1, prob[j])
  }
  return(list(Z, prob))
}


SL_update_Z_fast <- function(X, Y, alpha, Z, sigma2, lambda2, theta) {
  D <- dim(X)[1]; n <- dim(X)[2]; p <- dim(X)[3]
  Z_new <- Z
  prob  <- numeric(p)

  # 1) build initial M_d = I + X_S Lambda_S X_S^T and its chol factor L_d
  S      <- which(Z_new==1)
  M_list <- L_list <- vector("list", D)
  for(d in seq_len(D)) {
    if(length(S)>0) {
      Xd_S       <- matrix(X[d,, S], nrow=n)
      M_list[[d]] <- diag(n) + Xd_S %*% diag(lambda2[S], length(S)) %*% t(Xd_S)
    } else {
      M_list[[d]] <- diag(n)
    }
    L_list[[d]] <- chol(M_list[[d]])
  }

  # 2) flip one coordinate at a time, exactly as original SL_update_Z
  for (j in seq_len(p)) {
    # precompute the rank-1 vector u_d = sqrt(lambda2[j])*x_{d,j}
    U_list <- lapply(seq_len(D), function(d) sqrt(lambda2[j]) * X[d,,j])

    if (Z_new[j]==0) {
      # --- propose to INCLUDE j ---
      ll0 <- ll1 <- 0

      # (a) compute ll0 under current L_list
      for(d in seq_len(D)) {
        Ld  <- L_list[[d]]
        rd  <- Y[d,] - alpha[d]
        v   <- backsolve(Ld, rd,   transpose=TRUE)
        v   <- backsolve(Ld, v)
        quad0 <- crossprod(rd, v) / sigma2
        ll0   <- ll0 - 0.5*(2*sum(log(diag(Ld))) + quad0)
      }

      # (b) compute ll1 under M + u u^T (full chol)
      L1_list <- vector("list", D)
      for(d in seq_len(D)) {
        u    <- U_list[[d]]
        Md1  <- M_list[[d]] + tcrossprod(u)
        Ld1  <- chol(Md1)
        L1_list[[d]] <- Ld1

        rd   <- Y[d,] - alpha[d]
        v1   <- backsolve(Ld1, rd,   transpose=TRUE)
        v1   <- backsolve(Ld1, v1)
        quad1 <- crossprod(rd, v1) / sigma2
        ll1   <- ll1 - 0.5*(2*sum(log(diag(Ld1))) + quad1)
      }

      # (c) form posterior inclusion prob
      Rj      <- ll1 - ll0
      p1      <- theta[j] / (theta[j] + (1-theta[j])*exp(-Rj))
      prob[j] <- p1

      # (d) draw & commit
      if (rbinom(1,1,p1)==1) {
        Z_new[j] <- 1
        # update M_list and L_list
        for(d in seq_len(D)) {
          M_list[[d]] <- M_list[[d]] + tcrossprod(U_list[[d]])
          L_list[[d]] <-    L1_list[[d]]
        }
      }
      # else leave Z_new[j]==0, M_list, L_list unchanged

    } else {
      # --- propose to DROP j ---
      ll0 <- ll1 <- 0

      # (a) ll1 under current L_list
      for(d in seq_len(D)) {
        Ld  <- L_list[[d]]
        rd  <- Y[d,] - alpha[d]
        v   <- backsolve(Ld, rd,   transpose=TRUE)
        v   <- backsolve(Ld, v)
        quad1 <- crossprod(rd, v) / sigma2
        ll1   <- ll1 - 0.5*(2*sum(log(diag(Ld))) + quad1)
      }

      # (b) ll0 under M - u u^T
      L0_list <- vector("list", D)
      for(d in seq_len(D)) {
        u    <- U_list[[d]]
        Md0  <- M_list[[d]] - tcrossprod(u)
        Ld0  <- chol(Md0)
        L0_list[[d]] <- Ld0

        rd   <- Y[d,] - alpha[d]
        v0   <- backsolve(Ld0, rd,   transpose=TRUE)
        v0   <- backsolve(Ld0, v0)
        quad0 <- crossprod(rd, v0) / sigma2
        ll0   <- ll0 - 0.5*(2*sum(log(diag(Ld0))) + quad0)
      }

      # (c) posterior prob of keeping j out
      Rj      <- ll1 - ll0
      p1      <- theta[j] / (theta[j] + (1-theta[j])*exp(-Rj))
      prob[j] <- p1

      # (d) draw & commit
      if (rbinom(1,1,p1)==0) {
        Z_new[j] <- 0
        for(d in seq_len(D)) {
          M_list[[d]] <- M_list[[d]] - tcrossprod(U_list[[d]])
          L_list[[d]] <-    L0_list[[d]]
        }
      }
      # else leave Z_new[j]==1 unchanged
    }
  }

  list(Z = Z_new, prob = prob)
}







SL_update_theta = function(Z, beta_a = 1, beta_b = 1){
  sapply(1:length(Z), function(j){
    rbeta(1, beta_a + Z[j], beta_b + 1 - Z[j])
  })
}


SL_update_lambda2 = function(X, Y, alpha, Z, sigma2, lambda2, rho, prop_sd = 1){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  accept = rep(0, p)
  for(j in 1:p) if(Z[j]==1){
    # 1) save scalar
    t_old <- lambda2[j]

    # 2) propose scalar
    t_new <- rlnorm(1, meanlog=log(t_old), sdlog=prop_sd)

    # 3) log‐prior
    lp_old <- dgamma(t_old, shape=(D+1)/2, scale=2/(D*rho), log=TRUE)
    lp_new <- dgamma(t_new, shape=(D+1)/2, scale=2/(D*rho), log=TRUE)

    # 4) collapsed likelihood
    ll_old <- SL_collapse_likelihood(X, Y, alpha, Z, sigma2, lambda2, log=TRUE)

    # 5) overwrite in place and recompute
    lambda2[j] <- t_new
    ll_new <- SL_collapse_likelihood(X, Y, alpha, Z, sigma2, lambda2, log=TRUE)

    # 6) proposal densities
    q_fwd <- dlnorm(t_new, meanlog=log(t_old), sdlog=prop_sd, log=TRUE)
    q_bwd <- dlnorm(t_old, meanlog=log(t_new), sdlog=prop_sd, log=TRUE)

    # 7) MH ratio
    log_alpha <- (lp_new + ll_new + q_bwd) - (lp_old + ll_old + q_fwd)

    # 8) accept/reject and restore if needed
    if(log(runif(1)) < log_alpha) {
      accept[j] <- 1
      # keep lambda2[j] = t_new
    } else {
      accept[j] <- 0
      lambda2[j] <- t_old  # restore old value
    }
  } else {
    # Z[j]==0: conjugate draw
    lambda2[j] <- rgamma(1, shape=(D+1)/2, scale=2/(D*rho))
    accept[j]   <- 1
  }
  return(list(lambda2, accept))
}


SL_collapse_likelihood = function(X, Y, alpha, Z, sigma2, lambda2, log = TRUE){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  like_total = sapply(1:D, function(d){
    X_d = as.matrix(X[d,,])
    X_d_Z = X_d[,Z == 1,drop = FALSE]
    Y_d = Y[d,]
    alpha_d = alpha[d]
    Sigma_d = sigma2 * (diag(n) + X_d_Z %*% diag(lambda2[Z==1]) %*% t(X_d_Z))
    mvnfast::dmvn(Y_d, rep(alpha_d, n), Sigma_d, log = log)
  })

  if(log == TRUE){
    return(sum(like_total))
  }else{
    return(prod(like_total))
  }

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



#' Spike-and-Laplace MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using a spike-and-slab prior with a Laplace (double-exponential) slab
#' on nonzero coefficients. Latent inclusion indicators \code{Z} follow Bernoulli(\code{theta}), and their probabilities
#' follow independent Beta(\code{a}, \code{b}) priors. Local scales (\code{lambda2}) are updated via Metropolis-Hastings.
#'
#' @param X A 3D array of predictors with dimensions \code{D x n x p}.
#' @param Y A matrix of outcomes with dimensions \code{D x n}.
#' @param standardize Logical; whether to standardize \code{X} within each dataset. Default: \code{FALSE}.
#' @param a Shape parameter of Beta prior on inclusion probability. Default: \code{2}.
#' @param b Scale parameter of Beta prior (defaults to \code{2 * (D+1) / D} if not specified).
#' @param prop_sd Proposal standard deviation for log-normal Metropolis-Hastings updates of \code{lambda2}. Default: \code{1}.
#' @param nburn Number of burn-in iterations. Default: \code{4000}.
#' @param npost Number of posterior samples to store. Default: \code{4000}.
#' @param seed Optional random seed.
#' @param verbose Logical; whether to print MCMC progress. Default: \code{TRUE}.
#' @param printevery Print progress every \code{printevery} iterations. Default: \code{1000}.
#' @param chain_index Integer; index of the current MCMC chain (used for printing). Default: \code{1}.
#'
#' @return A list containing posterior draws of:
#' \describe{
#'   \item{post_rho}{\code{npost} vector of global scale parameters.}
#'   \item{post_Z}{\code{npost x p} matrix of binary inclusion indicators.}
#'   \item{post_inclusion_prob}{\code{npost x p} posterior inclusion probabilities (marginalized).}
#'   \item{post_theta}{\code{npost x p} matrix of inclusion probabilities per variable.}
#'   \item{post_alpha}{\code{npost x D} matrix of intercepts.}
#'   \item{post_lambda2}{\code{npost x p} matrix of local scale parameters.}
#'   \item{post_accept_ratio}{\code{p}-length vector of MH acceptance rates for each \code{lambda2_j}.}
#'   \item{post_sigma2}{\code{npost} vector of residual variances.}
#'   \item{post_beta}{\code{npost x D x p} array of regression coefficients.}
#'   \item{post_fitted_Y}{\code{npost x D x n} array of posterior predictive means (with noise).}
#'   \item{post_pool_beta}{\code{(npost * D) x p} matrix; reshaped \code{post_beta} for pooling.}
#'   \item{post_pool_fitted_Y}{\code{(npost * D) x n} matrix; reshaped \code{post_fitted_Y} for pooling.}
#'   \item{a}{Shape parameter used for the Beta prior.}
#'   \item{b}{Scale parameter used for the Beta prior.}
#'   \item{prop_sd}{Proposal SD used in MH update of \code{lambda2}.}
#' }
#'
#' @export
#'
#' @importFrom stats as.formula coef dgamma dlnorm gaussian glm lm median plogis predict quantile rbeta rbinom rcauchy residuals rgamma rlnorm rnorm runif sd var vcov

spike_laplace_mcmc = function(X, Y, standardize = F, a = 2, b = NULL, prop_sd = 1, nburn = 4000, npost = 4000, seed = NULL, verbose = T, printevery = 1000, chain_index = 1){
  if(!is.null(seed))
    set.seed(seed)
  if(standardize){
    X = aperm(simplify2array(lapply(1:dim(X)[1], function(i) scale(X[i, , ]))), c(3, 1, 2))
  }

  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  if(is.null(b))
    b = 2 * (D+1) / D

  pool = pooledResidualVariance(X, Y)
  sigma2 = pool$pooled_res_var
  beta = pool$beta
  alpha = pool$alpha
  theta = rbeta(p, a, b)
  Z = sapply(theta, function(t) rbinom(1, 1, t))
  rho = rgamma(1, shape = a, scale = b)
  lambda2 = rgamma(p, (D+1)/2, scale = 2 / (D * rho))




  post_rho = rep(NA, npost)
  post_Z = matrix(NA, npost, p)
  post_inclusion_prob = matrix(NA, npost, p)
  post_theta = matrix(NA, npost, p)
  post_alpha = matrix(NA, npost, D)
  post_lambda2 = matrix(NA, npost, p)
  post_accept = matrix(NA, nburn + npost, p)
  post_sigma2 = rep(NA, npost)
  post_beta = array(NA, dim = c(npost, D, p))
  post_fitted_Y = array(NA, dim = c(npost, D, n))

  for (i in 1:(nburn + npost)) {
    if(i %% printevery == 0 & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", ", ifelse(i <= nburn, "burn-in", "sampling"), sep = ""), "\n")
    if(i == (nburn + 1) & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", sampling", sep = ""), "\n")
    rho = SL_update_rho(lambda2, D, p, a, b)


    lambda2_list = SL_update_lambda2(X, Y, alpha, Z, sigma2, lambda2, rho, prop_sd = prop_sd)
    lambda2 = lambda2_list[[1]]
    accept = lambda2_list[[2]]
    Z_list = SL_update_Z_fast(X, Y, alpha, Z, sigma2, lambda2, theta)
    Z = Z_list[[1]]
    inclusion_prob = Z_list[[2]]
    theta = SL_update_theta(Z, 1, 1)
    alpha = SL_update_alpha(X, Y, Z, sigma2, lambda2)
    sigma2 = SL_update_sigma2(X, Y, alpha, Z, lambda2)
    beta = SL_update_beta(X, Y, alpha, Z, lambda2, sigma2)

    post_accept[i, ] = accept

    if(i > nburn){
      idx = i - nburn
      post_rho[idx] = rho
      post_Z[idx, ] = Z
      post_inclusion_prob[idx, ] = inclusion_prob
      post_theta[idx, ] = theta
      post_alpha[idx, ] = alpha
      post_lambda2[idx, ] = lambda2
      post_sigma2[idx] = sigma2
      post_beta[i - nburn, ,] = beta
      for (d in 1:D) {
        post_fitted_Y[idx, d, ] = X[d, , ] %*% beta[d, ] + alpha[d]+ rnorm(1,0,sqrt(sigma2))
      }
    }
  }

  return(list(
    post_rho = post_rho,
    post_Z = post_Z,
    post_inclusion_prob = post_inclusion_prob,
    post_theta = post_theta,
    post_alpha = post_alpha,
    post_lambda2 = post_lambda2,
    post_accept_ratio = colMeans(post_accept),
    post_sigma2 = post_sigma2,
    post_beta = post_beta,
    post_fitted_Y = post_fitted_Y,
    post_pool_beta = matrix(post_beta, nrow = dim(post_beta)[1] * dim(post_beta)[2],
                            ncol = dim(post_beta)[3]),
    post_pool_fitted_Y = matrix(post_fitted_Y, nrow = dim(post_fitted_Y)[1] * dim(post_fitted_Y)[2],
                                ncol = dim(post_fitted_Y)[3]),
    a = a, b = b, prop_sd = prop_sd
  ))
}
