SL_update_beta = function(X, Y, alpha, Z, lambda2, sigma2, hat_matrix){
  D = dim(X)[1]  # number of imputed datasets
  n = dim(X)[2]  # sample size per dataset
  p = dim(X)[3]  # number of predictors
  beta = matrix(0, nrow = D, ncol = p)  # initialize beta; inactive coefficients will remain zero
  hat_matrix_proj = array(0, dim = c(D, n, n))
  if(sum(Z) > 0){
    inv_variance = diag(1/lambda2[Z == 1])
    key <- paste0(as.integer(Z), collapse = "")
    if(key %in% names(hat_matrix)){
      XtX = hat_matrix[[key]]
      for(d in 1:D){
        va = Rfast::spdinv(XtX[d,,] + inv_variance)
        hat_matrix_proj[d,,] = as.matrix(X[d,,Z == 1]) %*% va %*% t(as.matrix(X[d,,Z == 1]))
        mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X[d,,Z == 1])) %*% va
        va = va * sigma2
        beta[d,Z == 1] = MASS::mvrnorm(1, mu, va)
      }
    }else{
      XtX = array(NA, dim = c(D, sum(Z), sum(Z)))
      for(d in 1:D){
        XtX[d,,] = t(as.matrix(X[d,,Z == 1])) %*% as.matrix(X[d,,Z == 1])
        va = Rfast::spdinv(XtX[d,,] + inv_variance)
        hat_matrix_proj[d,,] = as.matrix(X[d,,Z == 1]) %*% va %*% t(as.matrix(X[d,,Z == 1]))
        mu = (t(Y[d,] - alpha[d]) %*% as.matrix(X[d,,Z == 1])) %*% va
        va = va * sigma2
        beta[d,Z == 1] = MASS::mvrnorm(1, mu, va)
      }
      hat_matrix[[key]] = XtX
    }

  }
  return(list(beta = beta,  hat_matrix = hat_matrix, hat_matrix_proj = hat_matrix_proj))
}

SL_update_alpha = function(Y, Xbeta, sigma2) {
  D = dim(Y)[1]; n = dim(Y)[2]
  alpha = sapply(1:D, function(d){
    mu = mean(Y[d, ] - Xbeta[d, ])
    rnorm(1, mu, sqrt(sigma2 / n))
  })
  return(alpha)
}

# Update residual variance sigma2
SL_update_sigma2 = function(Y, Xbeta, beta_mul, alpha, Z, lambda2){
  D = dim(Y)[1]; n = dim(Y)[2]; p = length(beta_mul)
  a = D * (n + p) / 2
  SSE = sum(sapply(1:D, function(d) sum((Y[d, ] - Xbeta[d, ] - alpha[d])^2)))
  SSE_beta = sum(sapply(1:p, function(j) beta_mul[j] * (Z[j] == 1) / lambda2[j]))
  b = (SSE + SSE_beta) / 2
  return(MCMCpack::rinvgamma(1, a, b))
}


SL_update_rho = function(lambda2, D, p, a, b){
  shape = a + p * (D + 1) / 2
  scale = 1 / (1 / b + D * sum(lambda2) / 2)
  return(rgamma(1, shape, scale = scale))
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
      M_list[[d]] <- diag(n) + Xd_S %*% diag(lambda2[S]) %*% t(Xd_S)
    } else {
      M_list[[d]] <- diag(n)
    }
    L_list[[d]] <- chol(M_list[[d]])
  }

  # 2) flip one coordinate at a time, exactly as original SL_update_Z
  #print(Z_new)
  for (j in seq_len(p)) {
    #cat(j, " ", Z_new[j], "\n")
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



SL_update_Z = function(X, Y, alpha, Z, sigma2, lambda2, theta){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  prob = rep(1, p)
  for (j in 1:p) {
    #Z[j] = 1
    #R_j = SL_collapse_likelihood(X, Y, alpha, Z, sigma2, lambda2, log = TRUE)
    #Z[j] = 0
    #R_j = R_j - SL_collapse_likelihood(X, Y, alpha, Z, sigma2, lambda2, log = TRUE)
    R_j = logLikDiff_select(X, Y, lambda2, sigma2, Z, j)
    prob[j] = theta[j] / (theta[j] + (1 - theta[j]) * exp(-R_j))
    Z[j] = rbinom(1, 1, prob[j])
  }
  return(list(Z, prob))
}




logLikDiff_select <- function(X, Y, lambda, sigma2, s, j) {
  D = dim(X)[1]
  sum(sapply(1:D, function(d){
    X <- as.matrix(X[d,,])
    Y <- Y[d,]
    n <- nrow(X); p <- ncol(X)
    if (length(Y) != n) stop("length(Y) must equal nrow(X)")
    if (length(s) != p) stop("s must be length p = ncol(X)")

    if (is.matrix(lambda)) {
      if (!all(dim(lambda) == c(p,p))) stop("lambda must be p*p or length-p")
      lambda <- diag(lambda)
    } else if (length(lambda) != p) {
      stop("lambda must be length-p or p*p diag matrix")
    }

    sel_other <- which(s == 1 & seq_len(p) != j)
    X0   <- if (length(sel_other)) X[, sel_other, drop=FALSE] else matrix(0, n, 0)
    lam0 <- if (length(sel_other)) lambda[sel_other] else numeric(0)


    M0 <- diag(n)
    if (length(sel_other)) {
      M0 <- M0 + X0 %*% (lam0 * t(X0))
    }
    U0 <- chol(M0)
    logdet0 <- 2 * sum(log(diag(U0)))


    zY <- backsolve(U0, Y,    transpose = TRUE)
    wY <- backsolve(U0, zY,   transpose = FALSE)
    q0  <- crossprod(Y, wY)


    xj <- X[, j]
    zx <- backsolve(U0, xj,  transpose = TRUE)
    vj <- backsolve(U0, zx,  transpose = FALSE)
    s0 <- crossprod(xj, vj)
    t0 <- crossprod(Y,  vj)
    lambdaj <- lambda[j]


    alpha <- 1 + lambdaj * s0
    logdet_plus <- logdet0 + log(alpha)
    q_plus      <- q0 - (lambdaj * as.numeric(t0)^2) / alpha



    ll_minus <- -0.5 * (n * log(sigma2) + logdet0      + q0      / sigma2)
    ll_plus  <- -0.5 * (n * log(sigma2) + logdet_plus + q_plus / sigma2)

    ll_plus - ll_minus
  }))

}






SL_collapse_likelihood = function(X, Y, alpha, Z, sigma2, lambda2, log = TRUE){
  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  like_total = sapply(1:D, function(d){
    X_d = as.matrix(X[d,,])
    X_d_Z = X_d[,Z == 1, drop = FALSE]
    Y_d = Y[d,]
    alpha_d = alpha[d]

    if (ncol(X_d_Z) == 0) {
      Sigma_d = sigma2 * diag(n)
    } else {
      Sigma_d = sigma2 * (diag(n) + X_d_Z %*% diag(lambda2[Z == 1], ncol(X_d_Z)) %*% t(X_d_Z))
    }

    # Check positive-definiteness before evaluating likelihood
    ok = TRUE
    tryCatch({
      chol(Sigma_d)
    }, error = function(e) {
      ok <<- FALSE
    })

    if (!ok) {
      return(if (log) -1e10 else 0)
    }

    mvnfast::dmvn(Y_d, rep(alpha_d, n), Sigma_d, log = log)
  })

  if(log){
    return(sum(like_total))
  } else {
    return(prod(like_total))
  }
}




# Update local shrinkage lambda2 (per predictor)
SL_update_lambda2 = function(beta_mul, D, rho, sigma2, Z) {
  lambda2 = sapply(1:length(Z), function(i){
    if(Z[i] == 1)
      GIGrvg::rgig(1, 0.5, beta_mul[i] / sigma2, D * rho)
    else{
      rgamma(1, (D+1)/2, scale = 2 / (D * rho))
    }
  })
  return(lambda2)
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



#' Spike-and-Laplace MCMC Sampler for Multiply-Imputed Regression
#'
#' Implements Bayesian variable selection using a spike-and-slab prior with a Laplace (double-exponential) slab
#' on nonzero coefficients. Latent inclusion indicators \code{gamma} follow Bernoulli(\code{theta}), and their probabilities
#' follow independent Beta(\code{a}, \code{b}) priors.
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
#'   \item{\code{hat_matrix_proj}}{Matrix \code{D * n * n} of averaged projection hat-matrices. To avoid recalculate for estimating degree of freedom.}
#'   \item{\code{a}, \code{b}}{Numeric values of the rho hyperparameters used.}
#' }
#'
#' @examples
#' sim <- sim_B(n = 100, p = 20, type = "MAR", SNP = 1.5, corr = 0.5,
#' low_missing = TRUE, n_imp = 5, seed = 123)
#' X <- sim$data_MI$X
#' Y <- sim$data_MI$Y
#' fit <- spike_laplace_partially_mcmc(X, Y, nburn = 10, npost = 10)
#' @export
spike_laplace_partially_mcmc = function(X, Y, intercept = TRUE, a = 2, b = NULL, nburn = 4000, npost = 4000, seed = NULL, verbose = TRUE, printevery = 1000, chain_index = 1){
  if(!is.null(seed))
    set.seed(seed)

  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  if(is.null(b))
    b = (D+1) / (2 * D) / (a - 1)

  if(n > p){
    pool = pooledResidualVariance(X, Y, intercept)
    sigma2 = pool$pooled_res_var
    beta = pool$beta
    alpha = pool$alpha
  }else{
    pool = pooledResidualVariance(array(1, dim = c(D,n,1)), Y, intercept)
    sigma2 = pool$pooled_res_var
    beta = array(0, dim = c(D, p))#array(rnorm(D*p), dim = c(D, p))
    alpha = rep(0, D)
  }

  theta = rbeta(p, a, b)
  Z = sapply(theta, function(t) rbinom(1, 1, t))
  rho = rgamma(1, shape = a, scale = b)
  lambda2 = rgamma(p, (D+1)/2, scale = 2 / (D * rho))

  Xbeta = matrix(NA, D, n)
  for (d in 1:D) {
    Xbeta[d,] = X[d,,] %*% beta[d, ]
  }
  hat_matrix = list()

  hat_matrix_proj = array(0, dim = c(D, n, n))
  beta_mul = sapply(1:p, function(j) t(beta[,j]) %*% beta[,j])

  post_rho = rep(NA, npost)
  post_Z = matrix(NA, npost, p)
  post_inclusion_prob = matrix(NA, npost, p)
  post_theta = matrix(NA, npost, p)
  post_alpha = matrix(NA, npost, D)
  post_lambda2 = matrix(NA, npost, p)
  post_sigma2 = rep(NA, npost)
  post_beta = array(NA, dim = c(npost, D, p))
  post_fitted_Y = array(NA, dim = c(npost, D, n))

  for (i in 1:(nburn + npost)) {
    if(i %% printevery == 0 & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", ", ifelse(i <= nburn, "burn-in", "sampling"), sep = ""), "\n")
    if(i == (nburn + 1) & verbose)
      cat(paste("Chain", chain_index, ": ", i, "/", nburn + npost, ", sampling", sep = ""), "\n")
    rho = SL_update_rho(lambda2, D, p, a, b)

    theta = SL_update_theta(Z, 1, 1)

    lambda2 = SL_update_lambda2(beta_mul, D, rho, sigma2, Z)
    if(intercept)
      alpha = SL_update_alpha(Y, Xbeta, sigma2)
    sigma2 = SL_update_sigma2(Y, Xbeta, beta_mul, alpha, Z, lambda2)

    #start = Sys.time()
    Z_list = SL_update_Z(X, Y, alpha, Z, sigma2, lambda2, theta)
    Z = Z_list[[1]]
    inclusion_prob = Z_list[[2]]
    #end = Sys.time()
    #print(end - start)
    beta_list = SL_update_beta(X, Y, alpha, Z, lambda2, sigma2, hat_matrix)
    beta = beta_list[[1]]
    hat_matrix = beta_list[[2]]
    for (d in 1:D) {
      Xbeta[d,] = X[d,,] %*% beta[d, ]
    }
    beta_mul = sapply(1:p, function(j) t(beta[,j]) %*% beta[,j])

    if(i > nburn){
      hat_matrix_proj = hat_matrix_proj + beta_list[[3]]
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
        post_fitted_Y[idx, d, ] = X[d, , ] %*% beta[d, ] + alpha[d]+ stats::rnorm(n,0,sqrt(sigma2))
      }
    }
  }

  hat_matrix_proj = hat_matrix_proj / npost
  return(list(
    post_rho = post_rho,
    post_gamma = post_Z,
    #post_inclusion_prob = post_inclusion_prob,
    post_theta = post_theta,
    post_alpha = post_alpha,
    post_lambda2 = post_lambda2,
    post_sigma2 = post_sigma2,
    post_beta = post_beta,
    post_fitted_Y = post_fitted_Y,
    post_pool_beta = matrix(post_beta, nrow = dim(post_beta)[1] * dim(post_beta)[2],
                            ncol = dim(post_beta)[3]),
    post_pool_fitted_Y = matrix(post_fitted_Y, nrow = dim(post_fitted_Y)[1] * dim(post_fitted_Y)[2],
                                ncol = dim(post_fitted_Y)[3]),
    a = a, b = b, hat_matrix_proj = hat_matrix_proj
  ))
}
