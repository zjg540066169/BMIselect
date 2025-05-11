compute_bic_glm_fixed_sigma2 <- function(select, X, Y, sigma2) {
  n_thresh <- nrow(select)    # number of selection thresholds
  D <- dim(X)[1]              # number of imputations
  n <- dim(X)[2]              # sample size per imputation
  p <- dim(X)[3]
  N <- D * n                  # total observations

  bic_per_threshold <- numeric(n_thresh)

  for (t in seq_len(n_thresh)) {
    gamma_t <- select[t, ]
    selected_idx <- which(gamma_t == 1)
    k <- length(selected_idx)

    rss_total <- 0

    for (d in seq_len(D)) {
      yd <- Y[d, ]

      if (k == 0) {
        # Intercept-only model
        fit <- glm(yd ~ 1, family = gaussian(), weights = rep(1 / sigma2, n))
      } else {
        Xd_mat <- matrix(X[d, , selected_idx], nrow = n, ncol = k)
        Xd_df <- as.data.frame(Xd_mat)
        colnames(Xd_df) <- paste0("X", selected_idx)
        data_d <- data.frame(yd = yd, Xd_df)

        formula_str <- paste("yd ~", paste(colnames(Xd_df), collapse = " + "))
        fit <- glm(as.formula(formula_str), data = data_d,
                   family = gaussian(), weights = rep(1 / sigma2, n))
      }

      # Manually compute RSS: glm residuals are weighted, so use raw residuals
      y_hat <- predict(fit, type = "response")
      rss_total <- rss_total + sum((yd - y_hat)^2)
    }

    # BIC with fixed sigma²
    bic <- rss_total / sigma2 + (k + 1) * log(N)
    bic_per_threshold[t] <- bic
  }

  return(bic_per_threshold)
}







compute_normalized_bic_mi <- function(select, X, Y) {
  n_thresh <- nrow(select)    # number of threshold levels
  D <- dim(X)[1]              # number of imputations
  n <- dim(X)[2]
  p <- dim(X)[3]
  N <- D * n                  # total number of observations

  bic_per_threshold <- numeric(n_thresh)

  for (t in seq_len(n_thresh)) {
    gamma_t <- select[t, ]
    selected_idx <- which(gamma_t == 1)
    k <- length(selected_idx)

    rss_total <- 0

    for (d in seq_len(D)) {
      yd <- Y[d, ]

      if (k == 0) {
        fit <- lm(yd ~ 1)
      } else {
        Xd_mat <- matrix(X[d, , selected_idx], nrow = n, ncol = k)
        fit <- lm(yd ~ Xd_mat)
      }

      rss_total <- rss_total + sum(residuals(fit)^2)
    }

    # Normalized BIC per observation
    bic <- log(rss_total / N) + ((k + 1) * log(N)) / N
    bic_per_threshold[t] <- bic
  }

  return(bic_per_threshold)
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

fit_lr <- function(select, X, Y, intercept = TRUE) {
  X_i <- X[, select, drop = FALSE]
  if (intercept) {
    beta = rep(0, 1 + length(select))
    if(sum(beta) > 0){
      models <- lm(Y ~ X_i)
      beta[c(TRUE, select == TRUE)] = coef(models)
    }else{
      models <- lm(Y ~ 1)
      beta[1] = coef(models)
    }
  } else {
    beta = rep(0, length(select))
    if(sum(beta) > 0){
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

