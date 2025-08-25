#' Simulate dataset A: Independent continuous covariates with MCAR/MAR missingness
#'
#' Generates a dataset for Scenario A used in Bayesian MI-LASSO benchmarking. Covariates are iid standard normal,
#' with a fixed true coefficient vector, linear outcome, missingness imposed on specified columns under MCAR or MAR,
#' and multiple imputations via predictive mean matching.
#'
#' @param n Integer. Number of observations.
#' @param p Integer. Number of covariates (columns). Takes values in \{20, 40\}.
#' @param type Character. Missingness mechanism: "MCAR" or "MAR".
#' @param SNP Numeric. Signal-to-noise ratio controlling error variance.
#' @param low_missing Logical. If TRUE, use low missingness rates; if FALSE, higher missingness.
#' @param n_imp Integer. Number of multiple imputations to generate.
#' @param seed Integer or NULL. Random seed for reproducibility.
#'
#' @return A list with components:
#' \describe{
#'   \item{data_O}{A list of complete covariate matrix and outcomes before missingness.}
#'   \item{data_mis}{A list of covariate matrix and outcomes with missing values.}
#'   \item{data_MI}{A list of array of imputed covariates (n_imp × n × p) and a matrix of imputed outcomes (n_imp × n).}
#'   \item{data_CC}{A list of complete-case covariate matrix and outcomes.}
#'   \item{important}{Logical vector of true nonzero coefficient indices.}
#'   \item{covmat}{True covariance matrix used for X.}
#'   \item{beta}{True coefficient vector.}
#' }
#' @examples
#' sim <- sim_A(n = 100, p = 20, type = "MAR", SNP = 1.5,
#'              low_missing = TRUE, n_imp = 5, seed = 123)
#' str(sim)
#' @export
#' @importFrom stats rnorm rbinom complete.cases lm var sd median rgamma coef rcauchy rbeta vcov quantile
sim_A = function(n = 100, p = 20, type = "MAR", SNP = 1.5, low_missing = TRUE, n_imp = 5, seed = NULL){
  if(!is.null(seed))
    set.seed(seed)

  covmat = diag(p)
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = covmat)
  beta = rep(0, p)

  if(p == 20){
    beta[c(1,2,5,11,12,15)] = 1
  }else{
    beta[c(1,2,5,11,12,15,21,22,25,31,32,35)] = 1
  }

  sigma2 = t(beta) %*% covmat %*% beta / SNP
  Y = X %*% beta + rnorm(n, 0, sqrt(sigma2))

  # generate missingness
  if(low_missing == TRUE){
    if(type == "MCAR"){
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, 0.05)
        }
      }else{
        R = matrix(0, nrow = n, ncol = p)
        for (j in c(11:20, 31:40)) {
          R[,j] = rbinom(n, 1, 0.025)
        }
      }
    }else{
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, arm::invlogit(-3.4 + 0.5 * Y + 0.5 * X[, j - 10]))
        }
      }else{
        R = matrix(0, nrow = n, ncol = p)
        for (j in c(11:20, 31:40)) {
          R[,j] = rbinom(n, 1, arm::invlogit(-4.3 + 0.5 * Y + 0.5 * X[, j - 10]))
        }
      }
    }
  }else{
    if(type == "MCAR"){
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, 0.1)
        }
      }
    }else{
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, arm::invlogit(-2.1 + 0.5 * Y + 0.5 * X[, j - 10]))
        }
      }
    }
  }



  X_miss = as.matrix(X)
  X_miss[R == 1] = NA

  # imputation
  imp <- mice::mice(cbind(X_miss, Y), m = n_imp, method = "pmm", printFlag = FALSE)
  imp_list <- lapply(1:n_imp, function(i) as.matrix(mice::complete(imp, i)))
  imp_array <- array(0, dim = c(n_imp, n, p))
  for (d in 1:n_imp) {
    imp_array[d,,] = imp_list[[d]][,1:p]
  }

  X_O = list(X = X, Y = Y[,1])
  X_mis = list(X = X_miss, Y = Y[,1])
  X_MI = list(X = imp_array, Y = t(sapply(1:n_imp, function(i) Y)))
  X_CC = list(X = X_miss[complete.cases(X_miss),], Y = Y[complete.cases(X_miss)])
  return(list(
    data_O = X_O,
    data_mis = X_mis,
    data_MI = X_MI,
    data_CC = X_CC,
    important = (beta != 0),
    covmat = covmat,
    beta = beta
  ))
}





#' Simulate dataset B: AR(1)-correlated continuous covariates with MCAR/MAR missingness
#'
#' Generates a dataset for Scenario B used in Bayesian MI-LASSO benchmarking. Covariates are multivariate normal with AR(1) covariance,
#' with a fixed true coefficient vector, linear outcome, missingness imposed on specified columns under MCAR or MAR,
#' and multiple imputations via predictive mean matching.
#'
#' @param n Integer. Number of observations.
#' @param p Integer. Number of covariates (columns). Takes values in \{20, 40\}.
#' @param type Character. Missingness mechanism: "MCAR" or "MAR".
#' @param SNP Numeric. Signal-to-noise ratio controlling error variance.
#' @param low_missing Logical. If TRUE, use low missingness rates; if FALSE, higher missingness.
#' @param corr Numeric. AR(1) correlation parameter
#' @param n_imp Integer. Number of multiple imputations to generate.
#' @param seed Integer or NULL. Random seed for reproducibility.
#'
#' @return A list with components:
#' \describe{
#'   \item{data_O}{A list of complete covariate matrix and outcomes before missingness.}
#'   \item{data_mis}{A list of covariate matrix and outcomes with missing values.}
#'   \item{data_MI}{A list of array of imputed covariates (n_imp × n × p) and a matrix of imputed outcomes (n_imp × n).}
#'   \item{data_CC}{A list of complete-case covariate matrix and outcomes.}
#'   \item{important}{Logical vector of true nonzero coefficient indices.}
#'   \item{covmat}{True covariance matrix used for X.}
#'   \item{beta}{True coefficient vector.}
#' }
#' @examples
#' sim <- sim_B(n = 100, p = 20, type = "MAR", SNP = 1.5, corr = 0.5,
#'              low_missing = TRUE, n_imp = 5, seed = 123)
#' str(sim)
#' @export
sim_B = function(n = 100, p = 20, low_missing = TRUE, type = "MAR", SNP = 1.5, corr = 0.5, n_imp = 5, seed = NULL){
  if(!is.null(seed))
    set.seed(seed)

  covmat = AR(corr, p)
  # generate X, beta, alpha, Y
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = covmat)
  beta = rep(0, p)

  if(p == 20){
    beta[c(1,2,5,11,12,15)] = 1
  }else{
    beta[c(1,2,5,11,12,15,21,22,25,31,32,35)] = 1
  }


  sigma2 = t(beta) %*% covmat %*% beta / SNP
  Y = X %*% beta + rnorm(n, 0, sqrt(sigma2))

  # generate missingness
  if(low_missing == TRUE){
    if(type == "MCAR"){
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, 0.05)
        }
      }else{
        R = matrix(0, nrow = n, ncol = p)
        for (j in c(11:20, 31:40)) {
          R[,j] = rbinom(n, 1, 0.025)
        }
      }
    }else{
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, arm::invlogit(-3.6 + 0.5 * Y + 0.5 * X[, j - 10]))
        }
      }else{
        R = matrix(0, nrow = n, ncol = p)
        for (j in c(11:20, 31:40)) {
          alpha = 0
          R[,j] = rbinom(n, 1, arm::invlogit(-5.5 + 0.5 * Y + 0.5 * X[, j - 10]))
        }
      }
    }
  }else{
    if(type == "MCAR"){
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, 0.1)
        }
      }
    }else{
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, arm::invlogit(-1.9 + 0.5 * Y + 0.5 * X[, j - 10]))
        }
      }
    }
  }


  X_miss = as.matrix(X)
  X_miss[R == 1] = NA

  # imputation
  imp <- mice::mice(cbind(X_miss, Y), m = n_imp, method = "pmm", printFlag = FALSE)
  imp_list <- lapply(1:n_imp, function(i) as.matrix(mice::complete(imp, i)))
  imp_array <- array(0, dim = c(n_imp, n, p))
  for (d in 1:n_imp) {
    imp_array[d,,] = imp_list[[d]][,1:p]
  }

  X_O = list(X = X, Y = Y[,1])
  X_mis = list(X = X_miss, Y = Y[,1])
  X_MI = list(X = imp_array, Y = t(sapply(1:n_imp, function(i) Y)))
  X_CC = list(X = X_miss[complete.cases(X_miss),], Y = Y[complete.cases(X_miss)])
  return(list(
    data_O = X_O,
    data_mis = X_mis,
    data_MI = X_MI,
    data_CC = X_CC,
    important = (beta != 0),
    covmat = covmat,
    beta = beta
  ))
}






#' Simulate dataset C: AR(1)-latent Gaussian dichotomized to binary covariates with MCAR/MAR missingness
#'
#' Generates binary covariates by thresholding an AR(1) latent Gaussian, then proceeds as in sim_B.
#'
#' @param n Integer. Number of observations.
#' @param p Integer. Number of covariates (columns). Takes values in \{20, 40\}.
#' @param type Character. Missingness mechanism: "MCAR" or "MAR".
#' @param SNP Numeric. Signal-to-noise ratio controlling error variance.
#' @param low_missing Logical. If TRUE, use low missingness rates; if FALSE, higher missingness.
#' @param corr Numeric. AR(1) correlation parameter
#' @param n_imp Integer. Number of multiple imputations to generate.
#' @param seed Integer or NULL. Random seed for reproducibility.
#'
#' @return A list with components:
#' \describe{
#'   \item{data_O}{A list of complete covariate matrix and outcomes before missingness.}
#'   \item{data_mis}{A list of covariate matrix and outcomes with missing values.}
#'   \item{data_MI}{A list of array of imputed covariates (n_imp × n × p) and a matrix of imputed outcomes (n_imp × n).}
#'   \item{data_CC}{A list of complete-case covariate matrix and outcomes.}
#'   \item{important}{Logical vector of true nonzero coefficient indices.}
#'   \item{covmat}{True covariance matrix used for X.}
#'   \item{beta}{True coefficient vector.}
#' }
#' @examples
#' sim <- sim_C(n = 100, p = 20, type = "MAR", SNP = 1.5, corr = 0.5,
#'              low_missing = TRUE, n_imp = 5, seed = 123)
#' str(sim)
#' @export
sim_C = function(n = 100, p = 20, low_missing = TRUE, type = "MAR", SNP = 1.5, corr = 0.5, n_imp = 5, seed = NULL){
  if(!is.null(seed))
    set.seed(seed)

  covmat = AR(corr, p)
  # generate X, beta, alpha, Y
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = covmat)
  X = (X >= 0)
  beta = rep(0, p)

  if(p == 20){
    beta[c(1,2,5,11,12,15)] = 1
  }else{
    beta[c(1,2,5,11,12,15,21,22,25,31,32,35)] = 1
  }

  SigmaZ <- matrix(0, nrow = p, ncol = p)

  # fill in
  for (i in 1:p) {
    for (j in 1:p) {
      if (i == j) {
        SigmaZ[i, j] <- 1/4
      } else {
        SigmaZ[i, j] <- asin(corr^(abs(i - j))) / (2 * pi)
      }
    }
  }

  sigma2 = t(beta) %*% SigmaZ %*% beta / SNP
  Y = X %*% beta + rnorm(n, 0, sqrt(sigma2))

  # generate missingness
  if(low_missing == TRUE){
    if(type == "MCAR"){
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, 0.05)
        }
      }
    }else{
      R = matrix(0, nrow = n, ncol = p)
      for (j in 11:20) {
        R[,j] = rbinom(n, 1, arm::invlogit(-4.9 + 0.5 * Y + 0.5 * X[, j - 10]))
      }
    }
  }else{
    if(type == "MCAR"){
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, 0.1)
        }
      }
    }else{
      if(p == 20){
        R = matrix(0, nrow = n, ncol = p)
        for (j in 11:20) {
          R[,j] = rbinom(n, 1, arm::invlogit(-3.9 + 0.5 * Y + 0.5 * X[, j - 10]))
        }
      }
    }
  }

  X_miss = as.matrix(X)
  X_miss[R == 1] = NA

  # imputation
  imp <- mice::mice(cbind(X_miss, Y), m = n_imp, defaultMethod = "logreg", printFlag = FALSE)
  imp_list <- lapply(1:n_imp, function(i) as.matrix(mice::complete(imp, i)))
  imp_array <- array(0, dim = c(n_imp, n, p))
  for (d in 1:n_imp) {
    imp_array[d,,] = imp_list[[d]][,1:p]
  }

  X_O = list(X = X, Y = Y[,1])
  X_mis = list(X = X_miss, Y = Y[,1])
  X_MI = list(X = imp_array, Y = t(sapply(1:n_imp, function(i) Y)))
  X_CC = list(X = X_miss[complete.cases(X_miss),], Y = Y[complete.cases(X_miss)])
  return(list(
    data_O = X_O,
    data_mis = X_mis,
    data_MI = X_MI,
    data_CC = X_CC,
    important = (beta != 0),
    covmat = SigmaZ,
    beta = beta
  ))
}
