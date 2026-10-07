
compute_mi_bic <- function(X_arr, Y_arr, beta, df, alpha = NULL) {
  D     <- dim(X_arr)[1]
  n     <- dim(X_arr)[2]
  p     <- dim(X_arr)[3]
  stopifnot(
    all(dim(Y_arr)      == c(D, n)),
    all(dim(beta)  == c( D, p))
  )

  SSE <- 0
  for(d in seq_len(D)) {
    preds <- X_arr[d, , ] %*% beta[d, ]
    if(is.null(alpha))
      SSE   <- SSE + sum((Y_arr[d, ] - preds)^2)
    else
      SSE   <- SSE + sum((Y_arr[d, ] - alpha[d] - preds)^2)
  }
  # Pooled BIC_m: sample size D*n, effective projection degrees of freedom df.
  bic <- log(SSE / (D * n)) + df * log((D * n)) / (D * n)
  bic
}



#' Projecting Posterior Means of Full-Model Coefficients onto a Reduced Subset Model
#'
#' Given posterior means of \code{beta1_mat} (and optional intercepts
#' \code{alpha1_vec}) from a full model fitted on \code{D} imputed
#' datasets, compute the predictive projection onto the submodel defined by
#' \code{xs_vec}.  Returns the projected coefficients (and intercepts, if requested).
#'
#' @param X_arr A 3-D array of predictors, of dimension \code{D * n * p}.
#' @param beta1_mat A \code{D * p} matrix of full-model coefficients, one row per imputation.
#' @param xs_vec Logical vector of length \code{p}; \code{TRUE} for predictors to keep in the submodel.
#' @param sigma2 Numeric scalar; the residual variance from the full model (pooled across imputations).
#' @param alpha1_vec Optional numeric vector of length \code{D}; full-model intercepts per imputation.
#'   If \code{NULL} (the default), the projection omits an intercept term.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{beta2_mat}}{A \code{D * p} matrix of projected submodel coefficients.}
#'   \item{\code{alpha2_vec}}{(If \code{alpha1_vec} provided) numeric vector length \code{D} of projected intercepts.}
#' }
#'
#' @keywords internal
projection_mean <- function(X_arr,
                            beta1_mat,
                            xs_vec,
                            sigma2,
                            alpha1_vec = NULL) {
  # Dimensions
  D <- dim(X_arr)[1]
  n <- dim(X_arr)[2]
  p <- dim(X_arr)[3]

  # Input checks
  if (length(xs_vec) != p || !is.logical(xs_vec)) {
    stop("xs_vec must be a logical vector of length p")
  }

  ps <- sum(xs_vec)                # number of selected predictors
  beta2_mat <- matrix(0, nrow = D, ncol = p)
  if (!is.null(alpha1_vec)) {
    alpha2_vec <- numeric(D)
  }
  SS_j <- 0

  for (d in seq_len(D)) {
    Xd         <- X_arr[d, , , drop = TRUE]   # n * p
    b1         <- beta1_mat[d, ]              # length-p
    intercept1 <- if (is.null(alpha1_vec)) 0 else alpha1_vec[d]

    # Pseudo-response
    y_star <- Xd %*% b1 + intercept1         # length-n

    if (ps > 0) {
      Xs <- Xd[, xs_vec, drop = FALSE]       # n * ps

      if (is.null(alpha1_vec)) {
        # OLS without intercept
        XtX   <- crossprod(Xs)               # ps * ps
        XtY   <- crossprod(Xs, y_star)       # ps * 1
        coef_h <- solve(XtX, XtY)            # ps * 1

        beta2_mat[d, xs_vec] <- as.numeric(coef_h)
        resid <- y_star - Xs %*% coef_h

      } else {
        # OLS with intercept
        X_design <- cbind(1, Xs)              # n * (ps+1)
        XtX      <- crossprod(X_design)      # (ps+1) * (ps+1)
        XtY      <- crossprod(X_design, y_star)  # (ps+1) * 1
        coef_h   <- solve(XtX, XtY)          # (ps+1) * 1

        alpha2_vec[d]        <- coef_h[1]
        beta2_mat[d, xs_vec] <- as.numeric(coef_h[-1])
        resid <- y_star - X_design %*% coef_h
      }

    } else {
      # No predictors selected
      if (!is.null(alpha1_vec)) {
        alpha2_vec[d] <- mean(y_star)
        resid <- y_star - alpha2_vec[d]
      } else {
        resid <- y_star
      }
    }

    SS_j <- SS_j + sum(resid^2)
  }

  # Optimal variance
  sigma2_opt <- (n * D * sigma2 + SS_j) / (n * D)

  if (is.null(alpha1_vec)) {
    list(
      beta2_mat  = beta2_mat
    )
  } else {
    list(
      beta2_mat  = beta2_mat,
      alpha2_vec = alpha2_vec
    )
  }
}







#' Projection of Full-Posterior Draws onto a Reduced-Subset Model
#'
#' Given posterior draws \code{beta1_arr} (and optional intercepts \code{alpha1_arr})
#' from a full model fitted on \code{D} imputed datasets, compute
#' the predictive projection of each draw onto the submodel defined by \code{xs_vec}.
#' Returns the projected coefficients (and intercepts, if requested) plus the projected
#' residual variance for each posterior draw.
#'
#' @param X_arr A 3-D array of predictors, of dimension \code{D * n * p}.
#' @param beta1_arr A \code{npost * D * p} array of full-model coefficient draws.
#' @param sigma1_vec Numeric vector of length \code{npost}, full-model residual variances.
#' @param xs_vec Logical vector of length \code{p}; \code{TRUE} indicates predictors to keep.
#' @param alpha1_arr Optional \code{npost * D} matrix of full_model intercept draws.
#'   If \code{NULL} (the default), the projection omits an intercept term.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{beta2_arr}}{Array \code{npost * D * p} of projected submodel coefficients.}
#'   \item{\code{alpha2_arr}}{(If \code{alpha1_arr} provided) matrix \code{npost * D} of projected intercepts.}
#'   \item{\code{sigma2_opt}}{Numeric vector length \code{npost} of projected residual variances.}
#' }
#'
#' @keywords internal
projection_posterior <- function(X_arr,
                                 beta1_arr,
                                 sigma1_vec,
                                 xs_vec,
                                 alpha1_arr = NULL) {

  npost <- dim(beta1_arr)[1]
  D     <- dim(X_arr)[1]
  n     <- dim(X_arr)[2]
  p     <- dim(X_arr)[3]
  if(length(xs_vec) != p || !is.logical(xs_vec))
    stop("xs_vec must be a logical vector of length p")
  ps <- sum(xs_vec)
  has_int <- !is.null(alpha1_arr)

  # Prepare storage
  beta2_arr  <- array(0, dim = c(npost, D, p))
  if(has_int) {
    if(! (is.matrix(alpha1_arr) && all(dim(alpha1_arr)==c(npost, D))) ) {
      stop("alpha1_arr must be a matrix of dimension npost * D")
    }
    alpha2_arr <- matrix(0, nrow = npost, ncol = D)
  }
  SSacc <- numeric(npost)   # accumulated projection loss, per draw

  # Vectorized over draws: the projection operator depends only on the
  # imputation d (not the draw), so precompute it once per imputation and
  # apply it to all npost draws with a single matrix multiply.
  #   beta_s = M b,   M = (Xs'Xs)^{-1} Xs' X   [ps x p]
  #   ||resid||^2 = b' Q b,  Q = X'(I - Hs) X,  Hs = Xs (Xs'Xs)^{-1} Xs'
  for(d in seq_len(D)) {
    Xd    <- X_arr[d, , , drop = TRUE]              # n * p
    Bfull <- matrix(beta1_arr[, d, ], npost, p)     # npost * p

    if(ps > 0) {
      Xs <- Xd[, xs_vec, drop = FALSE]              # n * ps
      if(!has_int) {
        XtXinv <- solve(crossprod(Xs))
        M      <- XtXinv %*% crossprod(Xs, Xd)       # ps * p (precomputed once)
        beta2_arr[, d, xs_vec] <- Bfull %*% t(M)     # all draws at once
        Rop    <- Xd - Xs %*% M                       # (I - Hs) Xd
        Q      <- crossprod(Rop)                      # p * p
        SSacc  <- SSacc + rowSums((Bfull %*% Q) * Bfull)
      } else {
        Xdes   <- cbind(1, Xs)                        # n * (ps+1)
        DtDinv <- solve(crossprod(Xdes))
        Mfull  <- DtDinv %*% crossprod(Xdes, Xd)      # (ps+1) * p
        mint   <- as.numeric(DtDinv %*% crossprod(Xdes, rep(1, n)))
        int1   <- alpha1_arr[, d]
        coef   <- Bfull %*% t(Mfull) + outer(int1, mint)   # npost * (ps+1)
        alpha2_arr[, d]        <- coef[, 1]
        beta2_arr[, d, xs_vec] <- coef[, -1, drop = FALSE]
        resid  <- Bfull %*% t(Xd) + outer(int1, rep(1, n)) - coef %*% t(Xdes)
        SSacc  <- SSacc + rowSums(resid^2)
      }
    } else {
      # No predictors selected
      if(has_int) {
        int1  <- alpha1_arr[, d]
        ystar <- Bfull %*% t(Xd) + outer(int1, rep(1, n))
        mu    <- rowMeans(ystar)
        alpha2_arr[, d] <- mu
        SSacc <- SSacc + rowSums((ystar - mu)^2)
      } else {
        ystar <- Bfull %*% t(Xd)
        SSacc <- SSacc + rowSums(ystar^2)
      }
    }
  }

  # Optimal projected variance, per draw
  sigma2_opt <- (n * D * sigma1_vec + SSacc) / (n * D)

  # Return
  if(!has_int) {
    list(
      beta2_arr  = beta2_arr,    # npost * D * p
      sigma2_opt = sigma2_opt    # length npost
    )
  } else {
    list(
      beta2_arr  = beta2_arr,    # npost * D * p
      alpha2_arr = alpha2_arr,   # npost * D
      sigma2_opt = sigma2_opt    # length npost
    )
  }
}



#' Deficit-Calibrated Post-Selection Posterior for a Reduced-Subset Model
#'
#' Calibrates the (pushforward) projected posterior draws so that their
#' within-imputation covariance matches the submodel sampling covariance
#' \eqn{\hat\sigma_s^2 (X_s'X_s)^{-1}}, via a draw-by-draw "deficit" correction,
#' and returns residual-variance draws with the correct submodel degrees of
#' freedom \eqn{n - |s|}.  No model is refitted: the coefficient draws come from
#' the projection, and only the missing variance is added back.
#'
#' @param X_arr A 3-D array of predictors, dimension \code{D * n * p} (same scale
#'   used to produce \code{beta2_arr}; i.e. standardized when \code{standardize=TRUE}).
#' @param Y_arr A \code{D * n} matrix of responses (same scale; centered when
#'   \code{standardize=TRUE}).
#' @param beta2_arr A \code{npost * D * p} array of projected (pushforward)
#'   coefficient draws, e.g. \code{projection_posterior(...)$beta2_arr}.
#' @param xs_vec Logical vector length \code{p}; \code{TRUE} for selected predictors.
#' @param alpha1_arr Optional \code{npost * D} matrix of projected intercept draws
#'   (non-standardized case). If \code{NULL} (default) no intercept is used.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{beta_cal_arr}}{Array \code{npost * D * p} of calibrated coefficient draws.}
#'   \item{\code{sigma2_cal}}{Numeric vector length \code{npost * D} of calibrated
#'     residual-variance draws (pooled across imputations).}
#' }
#' @keywords internal
calibrate_posterior <- function(X_arr, Y_arr, beta2_arr, xs_vec, sigma2_draws = NULL, alpha1_arr = NULL) {
  npost   <- dim(beta2_arr)[1]
  D       <- dim(X_arr)[1]
  n       <- dim(X_arr)[2]
  p       <- dim(X_arr)[3]
  selidx  <- which(xs_vec)
  ps      <- length(selidx)
  has_int <- !is.null(alpha1_arr)
  nu      <- n - ps - as.integer(has_int)          # submodel residual df

  beta_cal_arr <- beta2_arr                         # start from pushforward draws
  sigma2_cal   <- numeric(npost * D)

  for (d in seq_len(D)) {
    Xd  <- X_arr[d, , , drop = TRUE]
    idx <- ((d - 1) * npost + 1):(d * npost)

    # nothing selected (or too few residual df): variance against intercept only
    if (ps == 0 || nu < 1) {
      mu  <- if (has_int) mean(alpha1_arr[, d]) else mean(Y_arr[d, ])
      rss <- sum((Y_arr[d, ] - mu)^2)
      sigma2_cal[idx] <- rss / stats::rchisq(npost, max(n - as.integer(has_int), 1))
      next
    }

    Xs <- Xd[, xs_vec, drop = FALSE]                # n * ps
    B  <- matrix(beta2_arr[, d, selidx], npost, ps) # projected coef draws
    m  <- colMeans(B)

    if (has_int) {
      Ainv <- solve(crossprod(cbind(1, Xs)))
      A    <- Ainv[-1, -1, drop = FALSE]            # beta block (accounts for intercept)
      rss  <- sum((Y_arr[d, ] - mean(alpha1_arr[, d]) - Xs %*% m)^2)
    } else {
      A    <- solve(crossprod(Xs))                  # (X_s'X_s)^{-1}
      yc   <- Y_arr[d, ] - mean(Y_arr[d, ])
      rss  <- sum((yc - Xs %*% m)^2)
    }
    # SHAPE-PRESERVING AFFINE calibration (per imputation d). Rescale the projected
    # draws so their covariance hits the flat-OLS target V = sbar * A EXACTLY, while
    # PRESERVING their non-normal shape (skew/heavy tails/spike). Center on
    # m = colMeans(B) so mean(beta_cal) == m EXACTLY -> per-imputation AND pooled
    # posterior means are UNCHANGED. The map is
    #   beta_cal = m + (B - m) %*% t(Cmap),   Cmap = V^{1/2} S^{-1/2},
    # so Cov(beta_cal) = Cmap S Cmap' = V^{1/2} (S^{-1/2} S S^{-1/2}) V^{1/2} = V.
    # (Contrast the old additive deficit, which convolved with Gaussian noise and
    # smoothed the projected shape toward normality.)
    S    <- stats::cov(B)                                # empirical cov of proj draws
    s2v  <- if (is.null(sigma2_draws)) rep(rss / nu, npost) else sigma2_draws
    sbar <- mean(s2v)                                    # mean projected sigma_s^2
    V    <- sbar * A                                     # flat-OLS target covariance
    eS   <- eigen(S, symmetric = TRUE); evS <- eS$values
    evS[evS < max(evS) * 1e-8] <- max(evS) * 1e-8        # ridge tiny/neg eigenvalues
    Sih  <- eS$vectors %*% diag(1 / sqrt(evS), ps) %*% t(eS$vectors)   # S^{-1/2}
    eV   <- eigen(V, symmetric = TRUE); evV <- pmax(eV$values, 0)
    Vh   <- eV$vectors %*% diag(sqrt(evV), ps) %*% t(eV$vectors)       # V^{1/2}
    Cmap <- Vh %*% Sih
    Bc   <- sweep(B, 2, m, "-") %*% t(Cmap)             # (B - m) %*% t(Cmap)
    beta_cal_arr[, d, selidx] <- sweep(Bc, 2, m, "+")  # + m  (mean preserved exactly)

    # calibrated residual-variance draws, correct submodel df
    sigma2_cal[idx] <- rss / stats::rchisq(npost, nu)
  }

  list(beta_cal_arr = beta_cal_arr, sigma2_cal = sigma2_cal)
}

