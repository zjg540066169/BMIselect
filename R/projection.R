
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
  #print(SSE)
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
#' @examples
#' # Simulate a single imputation with n=50, p=5:
#' D <- 3; n <- 50; p <- 5
#' X_arr <- array(rnorm(D * n * p), c(D, n, p))
#' beta1_mat <- matrix(rnorm(D * p), nrow = D)
#' # Suppose full-model sigma2 pooled is 1.2
#' sigma2 <- 1.2
#' # Project onto predictors 1 and 4 only:
#' xs_vec <- c(TRUE, FALSE, FALSE, TRUE, FALSE)
#' proj <- projection_mean(X_arr, beta1_mat, xs_vec, sigma2)
#' str(proj)
#'
#' # With intercept:
#' alpha1_vec <- rnorm(D)
#' proj2 <- projection_mean(X_arr, beta1_mat, xs_vec, sigma2, alpha1_vec)
#' str(proj2)
#'
#' @export
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
#' @examples
#' D <- 3; n <- 50; p <- 5; npost <- 100
#' X_arr      <- array(rnorm(D*n*p), c(D, n, p))
#' beta1_arr  <- array(rnorm(npost*D*p), c(npost, D, p))
#' sigma1_vec <- runif(npost, 0.5, 2)
#' xs_vec     <- c(TRUE, FALSE, TRUE, FALSE, TRUE)
#' # Without intercept
#' proj <- projection_posterior(X_arr, beta1_arr, sigma1_vec, xs_vec)
#' str(proj)
#' # With intercept draws
#' alpha1_arr <- matrix(rnorm(npost*D), nrow = npost, ncol = D)
#' proj2 <- projection_posterior(X_arr, beta1_arr, sigma1_vec, xs_vec, alpha1_arr)
#' str(proj2)
#' @export
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

  # Prepare storage
  beta2_arr  <- array(0, dim = c(npost, D, p))
  if(!is.null(alpha1_arr)) {
    if(! (is.matrix(alpha1_arr) && all(dim(alpha1_arr)==c(npost, D))) ) {
      stop("alpha1_arr must be a matrix of dimension npost * D")
    }
    alpha2_arr <- matrix(0, nrow = npost, ncol = D)
  }
  sigma2_opt <- numeric(npost)

  # Loop over posterior draws
  for(j in seq_len(npost)) {
    SS_j <- 0
    for(d in seq_len(D)) {
      Xd         <- X_arr[d, , , drop = TRUE]   # n * p
      b1         <- beta1_arr[j, d, ]           # length-p
      intercept1 <- if(is.null(alpha1_arr)) 0 else alpha1_arr[j, d]

      # Compute pseudo-response
      y_star <- Xd %*% b1 + intercept1        # length-n

      if(ps > 0) {
        Xs <- Xd[, xs_vec, drop = FALSE]      # n * ps

        if(is.null(alpha1_arr)) {
          # OLS WITHOUT intercept
          coef_h <- solve(crossprod(Xs), crossprod(Xs, y_star))
          beta2_arr[j, d, xs_vec] <- as.numeric(coef_h)
          resid <- y_star - Xs %*% coef_h
        } else {
          # OLS WITH intercept
          X_design <- cbind(1, Xs)             # n * (ps+1)
          coef_h   <- solve(crossprod(X_design),
                            crossprod(X_design, y_star))
          alpha2_arr[j, d]        <- coef_h[1]
          beta2_arr[j, d, xs_vec] <- as.numeric(coef_h[-1])
          resid <- y_star - X_design %*% coef_h
        }

      } else {
        # No predictors selected
        if(!is.null(alpha1_arr)) {
          alpha2_arr[j, d] <- mean(y_star)
          resid <- y_star - alpha2_arr[j, d]
        } else {
          resid <- y_star
        }
      }

      SS_j <- SS_j + sum(resid^2)
    }

    # Optimal projected variance
    sigma2_opt[j] <- (n * D * sigma1_vec[j] + SS_j) / (n * D)
  }

  # Return
  if(is.null(alpha1_arr)) {
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

