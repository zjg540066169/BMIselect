find_beta2_sigma2_array <- function(X_arr, beta1_mat, alpha1_vec,  sigma1_sq, xs_vec) {
  # dimensions
  D <- dim(X_arr)[1]
  n <- dim(X_arr)[2]
  p <- dim(X_arr)[3]
  # input checks
  if(length(alpha1_vec) != D) {
    stop("alpha1_vec must be numeric vector of length D")
  }
  if(length(xs_vec) != p || !is.logical(xs_vec)) {
    stop("xs_vec must be a logical vector of length p")
  }
  # number of selected predictors
  ps <- sum(xs_vec)

  # initialize storage
  alpha2_vec <- numeric(D)
  beta2_mat  <- matrix(0, nrow = D, ncol = p)
  SS <- 0

  for(d in seq_len(D)) {
    # extract data for imputation d
    X  <- X_arr[d, , , drop = TRUE]    # n × p
    b1 <- beta1_mat[d, ]               # p-vector
    a1 <- alpha1_vec[d]                # scalar intercept

    # pseudo-response from the full model
    y_star <- a1 + X %*% b1             # n-vector

    # subset design matrix
    Xs <- X[, xs_vec, drop = FALSE]     # n × ps

    # add intercept column for subset model
    Xs_aug <- cbind(Intercept = 1, Xs)  # n × (ps + 1)

    # closed-form OLS: solve for coefficients
    XtX   <- crossprod(Xs_aug)          # (ps+1) × (ps+1)
    XtY   <- crossprod(Xs_aug, y_star)  # (ps+1) × 1
    coeffs <- solve(XtX, XtY)           # (ps+1) × 1

    # store parameters
    alpha2_vec[d]            <- coeffs[1]
    beta2_mat[d, xs_vec]     <- as.numeric(coeffs[-1])

    # accumulate residual sum of squares
    resid <- y_star - Xs_aug %*% coeffs  # n-vector
    SS <- SS + sum(resid^2)
  }

  # compute KL-optimal common variance
  sigma2_sq <- (n * D * sigma1_sq + SS) / (n * D)

  list(
    alpha2_vec = alpha2_vec,
    beta2_mat  = beta2_mat,
    sigma2_sq  = as.numeric(sigma2_sq)
  )
}


# Sensitivity (Recall): proportion of truly important variables that are selected.
sensitivity <- function(truth, select) {
  if(is.null(dim(select)))
    sum(truth & select) / sum(truth)
  else{
    apply(select, 1, function(x) sum(truth & x) / sum(truth))
  }
}

# Specificity: proportion of truly unimportant variables that are not selected.
specificity <- function(truth, select) {
  if(is.null(dim(select)))
    sum(!truth & !select) / sum(!truth)
  else{
    apply(select, 1, function(x) sum(!truth & !x) / sum(!truth))
  }
}

# Precision: proportion of selected variables that are truly important.
precision <- function(truth, select) {
  if(is.null(dim(select)))
    sum(truth & select) / sum(select)
  else{
    apply(select, 1, function(x) sum(truth & x) / sum(x))
  }
}

# Recall: same as sensitivity.
AR = function (rho, p)
{
  sigma <- matrix(0, nrow = p, ncol = p)
  for (i in 1:p) {
    for (j in 1:i) {
      sigma[i, j] <- sigma[j, i] <- rho^(i - j)
    }
  }
  return(sigma)
}

recall <- function(truth, select) {
  if(is.null(dim(select)))
    sum(truth & select) / sum(truth)
  else{
    apply(select, 1, function(x) sum(truth & x) / sum(truth))
  }
}

# F1-score: harmonic mean of precision and recall.
f1_score <- function(truth, select) {
  pre <- precision(truth, select)
  sen <- sensitivity(truth, select)

  f1 = 2 * pre * sen / (pre + sen)
  f1[is.nan(f1)] = 0
  f1
}


sign_acc = function(true_coefficients, select, X, Y){
  if(length(dim(X)) == 2){
    X_ = array(NA, dim = c(1, dim(X)[1], dim(X)[2]))
    X_[1,,] = X
    Y_ = matrix(NA, nrow = 1, length(Y))
    Y_[1,] = Y
    X = X_
    Y = Y_
  }
  acc = apply(select, 1, function(s){
    mean(sign(pooled_coefficients(s, X, Y)[-1]) == sign(true_coefficients))
  })
  names(acc) = rownames(select)
  acc
}


sign_acc_beta = function(true_coefficients, coefficients){
  if(is.null(dim(coefficients))){
    coefficients = t(as.matrix(coefficients))
  }
  acc = apply(coefficients, 1, function(s){
    mean(sign(s) == sign(true_coefficients))
  })
  acc
}



fit_lr <- function(select, X, Y, intercept = TRUE) {
  X_i <- X[, select, drop = FALSE]
  if (intercept) {
    beta = rep(0, 1 + length(select))
    if(sum(select) > 0){
      models <- lm(Y ~ X_i)
      beta[c(TRUE, select == TRUE)] = coef(models)
    }else{
      models <- lm(Y ~ 1)
      beta[1] = coef(models)
    }
  } else {
    beta = rep(0, length(select))
    if(sum(select) > 0){
      models <- lm(Y ~ - 1 + X_i)
      beta[select == TRUE] = coef(models)
    }
  }
  return(beta)
}

# pooled_coefficients: returns pooled coefficient estimates (via simple averaging).
pooled_coefficients <- function(select, X, Y, intercept = TRUE) {
  m <- dim(X)[1]
  coefs <- vector("list", m)

  for (i in 1:m) {
    X_i <- X[i, , ]
    X_i <- X_i[, select, drop = FALSE]
    X_i <- as.matrix(X_i)

    if (intercept) {
      beta = rep(0, 1 + length(select))
      if(sum(select) > 0){
        models <- lm(Y[i, ] ~ X_i)
        beta[c(TRUE, select == TRUE)] = coef(models)
      }else{
        models <- lm(Y[i, ] ~ 1)
        beta[1] = coef(models)
      }
    } else {
      beta = rep(0, length(select))
      if(sum(select) > 0){
        models <- lm(Y[i, ] ~ - 1 + X_i)
        beta[select == TRUE] = coef(models)
      }
    }
    coefs[[i]] = beta
  }
  # Average the coefficients over imputations.
  pooled <- Reduce("+", coefs) / m
  return(pooled)
}






# pooled_covariance: returns the pooled covariance matrix using Rubin’s rules.
pooled_covariance <- function(select, X, Y, intercept = TRUE) {
  m <- dim(X)[1]
  cov_list <- vector("list", m)
  beta_outer <- vector("list", m)

  for (i in 1:m) {
    X_i <- X[i, , ]
    X_i <- X_i[, select, drop = FALSE]
    X_i <- as.matrix(X_i)

    if (intercept) {
      models <- lm(Y[i, ] ~ X_i)
    } else {
      models <- lm(Y[i, ] ~ - 1 + X_i)
    }

    cov_list[[i]] <- vcov(models)
    beta_i <- coef(models)
    beta_outer[[i]] <- beta_i %o% beta_i  # outer product
  }

  within <- Reduce("+", cov_list) / m
  between <- Reduce("+", beta_outer) / (m - 1)
  pooled_cov <- within + (1 + 1/m) * between
  return(pooled_cov)
}

# mse: calculates mean-squared error using pooled coefficients and a given true beta.
# beta: the true coefficient vector for the selected variables.
# covariance: the pooled covariance matrix corresponding to the selected variables.
# If intercept==TRUE, the pooled coefficients include an intercept which we discard.
mse <- function(beta, covariance, select, X, Y, intercept = TRUE) {
  if (sum(select) == 0) {
    pool_beta <- rep(0, length(beta))
  } else {
    m <- dim(X)[1]
    coefs <- vector("list", m)
    for (i in 1:m) {
      X_i <- X[i, , ]
      X_i <- X_i[, select, drop = FALSE]
      X_i <- as.matrix(X_i)

      if (intercept) {
        models <- lm(Y[i, ] ~ X_i)
      } else {
        models <- lm(Y[i, ] ~ - 1 + X_i)
      }
      coefs[[i]] <- coef(models)
    }
    pool_coef <- Reduce("+", coefs) / m
    if (!intercept) {
      # Remove the intercept; assume true beta corresponds only to predictors.
      pool_beta <- pool_coef[-1]
    } else {
      pool_beta <- pool_coef
    }
  }
  print(pool_beta)
  print(beta)
  diff <- pool_beta - beta
  mse_val <- as.numeric(t(diff) %*% covariance %*% (diff))
  return(mse_val)
}


coverage = function(x, truth, alpha = 0.05){
  mean(sapply(1:length(truth), function(j){
    (quantile(x[,j], alpha / 2) <= truth[j]) & (quantile(x[,j], 1 - alpha / 2) >= truth[j])
  }))
}
