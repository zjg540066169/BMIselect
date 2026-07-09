if (getRversion() >= "2.15.1") {
  utils::globalVariables("lambda")
}

#' Multiple-Imputation LASSO (MI-LASSO)
#'
#' Fit a LASSO-like penalty across \code{D} multiply-imputed datasets by
#' iteratively reweighted ridge regressions (Equation (4) of the manuscript).
#' For each tuning parameter in \code{lamvec}, it returns the pooled
#' coefficient estimates, the BIC, and the selected variables.
#'
#' @param X A matrix \code{n×p} or an array \code{D×n×p} of imputed predictor sets.
#'   If a matrix is supplied, it is treated as a single imputation (\code{D = 1}).
#' @param Y A vector length \code{n} or a \code{D×n} matrix of outcomes.  If a
#'   vector, it is reused across imputations.
#' @param lamvec Numeric vector of penalty parameters \eqn{\lambda} to search. Default \code{(2^(seq(-1,4,by=0.05)))^2/2}.
#' @param maxiter Integer; maximum number of ridge–update iterations per \code{lambda}.
#'   Default \code{200}.
#' @param eps Numeric; convergence tolerance on coefficient change. Default \code{1e-20}.
#' @param ncores Integer; number of cores for parallelizing over \code{lamvec}.
#'   Default \code{1}.
#'
#' @return If \code{length(lamvec) > 1}, a list with elements:
#'   \describe{
#'     \item{\code{best}}{List for the \eqn{lambda} with minimal BIC containing:
#'       \code{coefficients} (\code{(p+1)×D} intercept + slopes),
#'       \code{bic} (BIC scalar),
#'       \code{varsel} (logical length-\code{p} vector of selected predictors),
#'       \code{lambda} (the chosen penalty).}
#'     \item{\code{lambda_path}}{\code{length(lamvec)×2} matrix of each
#'       \code{lambda} and its corresponding BIC.}
#'   }
#'   If \code{length(lamvec) == 1}, returns a single list (as above) for that
#'   penalty.
#'
#' @examples
#' sim <- sim_A(n = 100, p = 20, type = "MAR", SNP = 1.5, low_missing = TRUE, n_imp = 5, seed = 123)
#' X <- sim$data_MI$X
#' Y <- sim$data_MI$Y
#' fit <- MI_LASSO(X, Y, lamvec = c(0.1))
#' @export
MI_LASSO = function(X, Y, lamvec = (2^(seq(-1,4,by=0.05)))^2/2, maxiter=200, eps=1e-20, ncores = 1) {
  start = Sys.time()
  ## D is the number of imputations
  ## mydata is in the array format: mydata[[1]] is the first imputed dataset...
  ## for each mydata[[d]], the first p columns are covariates X, and the last one is the outcome Y

  if(is.null(dim(X))){
    X = array(X, dim = c(1, length(X), 1))
  }else if(length(dim(X)) == 2){
    X = array(X, dim = c(1, nrow(X), ncol(X)))
  }

  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]


  if(is.null(dim(Y))){
    Y = t(sapply(1:D, function(i) Y))
  }
  if(dim(Y)[2] == 1){
    Y = t(sapply(1:D, function(i) Y[,1]))
  }

  ## Standardize covariates X and center outcome Y
  x = NULL
  y = NULL
  meanx = NULL
  normx = NULL
  meany = 0
  for (d in 1:D) {
    x[[d]] = X[d,,]
    y[[d]] = Y[d,]
    meanx[[d]] = apply(x[[d]], 2, mean)
    x[[d]] = scale(x[[d]], meanx[[d]], FALSE)
    normx[[d]] = sqrt(apply(x[[d]]^2, 2, sum))
    x[[d]] = scale(x[[d]], FALSE, normx[[d]])
    meany[d] = mean(y[[d]])
    y[[d]] = y[[d]] - meany[d]
  }

  ## Ordinary least squares (OLS) estimates of beta coefficients
  b.ols = matrix(0,p,D)
  for (d in 1:D) {
    b.ols[,d]  = qr.solve(t(x[[d]])%*%x[[d]])%*%t(x[[d]])%*%y[[d]]
  }

  if(ncores > 1 & length(lamvec) > 1){
    doParallel::registerDoParallel(ncores)
  }else{
    foreach::registerDoSEQ()
  }
  xtx = list()
  xty = list()
  for (d in 1:D) {
    xtx[[d]] = t(x[[d]])%*%x[[d]]
    xty[[d]] = t(x[[d]])%*%y[[d]]
  }

  `%dopar%` <- foreach::`%dopar%`
  model_lambda = suppressWarnings(foreach::foreach(lambda = lamvec, .combine = list, .multicombine = TRUE, .maxcombine = ifelse(length(lamvec) >= 2, length(lamvec), 2)) %dopar% {
    ## Estimate beta_dj in Equation (4)
    iter = 0
    dif = 1
    c = rep(1,p)
    b = matrix(0,p,D)
    while (dif>=eps & iter < maxiter) {
      iter = iter + 1
      b.old = b
      # update beta_d by ridge regression
      for (d in 1:D) {
        xtx_temp = xtx[[d]]
        diag(xtx_temp) = diag(xtx_temp) + lambda/c
        b[,d] = qr.solve(xtx_temp)%*%xty[[d]]
      }
      # update c in Equation (4)
      c = sqrt(apply(b^2,1,sum))
      c[c<sqrt(D)*1e-10] = sqrt(D)*1e-10
      dif = max(abs(b-b.old))
    }
    b[apply((b^2),1,sum)<=5*1e-8,] = 0

    ## Calculate BIC
    sse = 0
    df = 0
    nzero = seq(p)[apply(b^2, 1, sum)>0]
    for (d in 1:D) {
      sse[d] = mean((y[[d]]-x[[d]]%*%b[,d])^2)
    }
    for (j in nzero) {
      norm1 = sqrt(sum(b[j,]^2))
      norm2 = sqrt(sum(b.ols[j,]^2))
      df = df+1+(D-1)*norm1/norm2
    }
    bic = log(mean(sse)) + log(D*n)*df/(D*n)

    ## transform beta and intercept back to original scale
    b.scaled = b
    b0 = 0
    coefficients = matrix(0,p+1,D)
    for (d in 1:D) {
      b[,d] = b[,d]/normx[[d]]
      b0[d] = meany[d] - b[,d]%*%meanx[[d]]
      coefficients[,d] = c(b0[d], b[,d])
    }

    ## output the selected variables
    ## TRUE means "selected", FALSE means "NOT selected"
    varsel = abs(b[,1])>0

    return(list(coefficients=coefficients, bic=bic, varsel=varsel, lambda = lambda))
  })

  end <- Sys.time()

  cat(sprintf("Running time for %d %s: %.2f minutes\n",
              ncores, ifelse(ncores > 1, "ncore", "ncores"),
              as.numeric(difftime(end, start, units = "mins"))))



  if(length(lamvec) > 1){
    doParallel::stopImplicitCluster()
    foreach::registerDoSEQ()

    return(list(
      best = model_lambda[[which.min(do.call(c, lapply(model_lambda, function(m) m$bic)))]],
      lambda_path = cbind(lamvec, do.call(c, lapply(model_lambda, function(m) m$bic)))
    ))
  }else{
    return(model_lambda)
  }
}
