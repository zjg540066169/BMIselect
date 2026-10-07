if (getRversion() >= "2.15.1") {
  utils::globalVariables(c("i", "ij"))
}

# Rank-normalized split-Rhat / ESS per coefficient, computed ACROSS chains on the
# pooled draws with the D imputations INTERLEAVED (imputation fastest-varying) in the
# per-chain draw order.
#
# Stacking the D imputations in BLOCKS along the draw axis
# (matrix(post_beta[npost,D,p], npost*D, p)) makes split-Rhat compare halves with
# different imputation composition and structurally inflates it even for a converged
# sampler (e.g. Rhat ~1.16 when the interleaved value is ~1.00). Interleaving keeps
# every contiguous half balanced over imputations, so the standard rank-normalized
# split-Rhat (Vehtari, Gelman, Simpson, Carpenter & Buerkner 2021) applies directly
# to the pooled coefficient, using all chains jointly. `arrs` is a length-nchains
# list of [npost, D, V] arrays (V = coefficients; sigma2 uses D = V = 1). Returns one
# value per coefficient.
.rhat_ess_pooled <- function(arrs) {
  d <- dim(arrs[[1]]); np <- d[1]; Dd <- d[2]; V <- d[3]
  rh <- eb <- et <- rep(NA_real_, V)
  for (v in seq_len(V)) {
    # per chain: interleave to [t1d1, t1d2, ..., t1dD, t2d1, ...]  (length np*Dd),
    # then M is (np*Dd) x nchains -> one rank-normalized split-Rhat across chains.
    M <- vapply(arrs, function(a) as.vector(t(matrix(a[, , v], np, Dd))), numeric(np * Dd))
    if (all(is.finite(M)) && stats::sd(as.numeric(M)) > 0) {
      rh[v] <- suppressWarnings(posterior::rhat(M))
      eb[v] <- suppressWarnings(posterior::ess_bulk(M))
      et[v] <- suppressWarnings(posterior::ess_tail(M))
    }
  }
  list(rhat = rh, ess_bulk = eb, ess_tail = et)
}

# Pooled multi-chain SELECTION (steps 1-4 of the four-step, on the pooled draws):
# concatenate every chain's draws into one pseudo-chain and run candidate generation
# -> rank cap -> BIC/LOO scoring ONCE, returning the single shared best_select (plus
# the pooled pseudo-chain pc and the candidate/score objects). Used BOTH to project
# every chain onto the SAME submodel -- so the selected-model split-Rhat measures
# sampler convergence on a common support, not cross-chain disagreement about which
# variables each chain happened to select -- AND by .pooled_multichain() below for
# the pooled selected-model posterior.
.pooled_select <- function(X, Y, model_chains, model, standardize, grid,
                           selection_set, criterion, search) {
  nchains <- length(model_chains)
  D <- dim(X)[1]; n <- dim(X)[2]; p <- dim(X)[3]
  NP <- dim(model_chains[[1]]$post_beta)[1] * nchains

  # ---- 1. pooled pseudo-chain: concatenate draws along the draw axis; the per-
  #        chain hat_matrix_proj is already a posterior mean, so average over chains.
  pc <- list()
  pc$post_beta   <- abind::abind(lapply(model_chains, function(ch) ch$post_beta), along = 1)
  pc$post_sigma2 <- unlist(lapply(model_chains, function(ch) ch$post_sigma2))
  pc$post_pool_beta <- matrix(pc$post_beta, nrow = NP * D, ncol = p)  # column summaries are order-invariant
  if (!is.null(model_chains[[1]]$post_alpha))
    pc$post_alpha <- abind::abind(lapply(model_chains, function(ch) ch$post_alpha), along = 1)
  if (!is.null(model_chains[[1]]$post_gamma))
    pc$post_gamma <- do.call(rbind, lapply(model_chains, function(ch) ch$post_gamma))
  pc$hat_matrix_proj <- Reduce(`+`, lapply(model_chains, function(ch) ch$hat_matrix_proj)) / nchains

  bic_models <- NULL; loo_models <- NULL; select <- NULL
  # ---- 2. candidate generation (mirrors step 10) ----
  if (!is.null(selection_set)) {
    best_select <- matrix(as.logical(selection_set), nrow = 1)
  } else {
    if (search == "forward") {
      select <- forward_candidates(X, list(pc), n_max = n - 2, standardize = standardize)[[1]]
    } else if (search == "snc") {
      SNCv <- apply(pc$post_pool_beta, 2, function(x) mean(abs(x) <= sqrt(stats::var(x))))
      select <- unique(rbind(
        t(sapply(as.character(grid), function(ci) SNCv < as.numeric(ci), simplify = TRUE, USE.NAMES = TRUE)),
        "median" = (apply(pc$post_pool_beta, 2, stats::median) != 0)))
    } else if (model %in% c("Multi_Laplace", "Horseshoe", "ARD")) {
      select <- unique(t(sapply(as.character(grid), function(ci) {
        ci <- as.numeric(ci)
        apply(pc$post_pool_beta, 2, function(x_j) prod(sign(stats::quantile(x_j, c((1 - ci)/2, 1 - (1 - ci)/2)))))
      }, simplify = TRUE, USE.NAMES = TRUE)) == 1)
    } else {
      select <- rbind(
        "median" = (apply(pc$post_pool_beta, 2, stats::median) != 0),
        unique(t(sapply(as.character(grid), function(ci)
          apply(pc$post_gamma, 2, function(x_j) mean(x_j) >= as.numeric(ci)),
          simplify = TRUE, USE.NAMES = TRUE))))
      select[, colMeans(pc$post_gamma) == 1] <- TRUE
      select[, colMeans(pc$post_gamma) == 0] <- FALSE
    }
    # ---- 3. rank cap (mirrors step 11) ----
    full_ok <- sapply(1:D, function(d) qr(cbind(1, X[d, , , drop = TRUE]))$rank == p + 1)
    valid <- sapply(1:nrow(select), function(i) {
      sc <- which(select[i, ])
      if (length(sc) > n - 2) return(FALSE)
      if (length(sc) == 0)     return(TRUE)
      all(sapply(1:D, function(d) if (full_ok[d]) TRUE else
        qr(cbind(1, X[d, , , drop = TRUE][, sc, drop = FALSE]))$rank == length(sc) + 1))
    })
    select <- select[valid, , drop = FALSE]
    # ---- 4. model-size selection (BIC default; non-orthogonal projection df) ----
    if (criterion == "bic") {
      beta_mean <- apply(pc$post_beta, c(2, 3), mean); s2_mean <- mean(pc$post_sigma2)
      bic_models <- apply(select, 1, function(subselect) {
        pbs <- if (standardize) projection_mean(X, beta_mean, subselect, s2_mean)
               else             projection_mean(X, beta_mean, subselect, s2_mean, colMeans(pc$post_alpha))
        df <- if (sum(subselect) == 0) 0 else sum(sapply(1:D, function(d) {
          X_ds <- X[d, , ][, subselect, drop = FALSE]
          sum(diag(Rfast::spdinv(crossprod(X_ds)) %*% crossprod(X_ds, pc$hat_matrix_proj[d, , ] %*% X_ds)))
        }))
        if (standardize) c(BIC = compute_mi_bic(X, Y, pbs$beta2_mat, df = df), df = df)
        else             c(BIC = compute_mi_bic(X, Y, pbs$beta2_mat, df = df, alpha = pbs$alpha2_vec), df = df)
      })
      best_select <- select[which.min(bic_models[1, ]), , drop = FALSE]
    } else {
      loo_models  <- loo_select(X, Y, list(pc), list(select), standardize = standardize)[[1]]
      idx         <- if (criterion == "loo-1se") loo_models$best_1se else loo_models$best
      best_select <- select[idx, , drop = FALSE]
    }
  }
  list(best_select = best_select, select = select, bic_models = bic_models,
       loo_models = loo_models, pc = pc)
}

# Additive multi-chain POOLED posterior (steps 5-7): given the shared selection from
# .pooled_select(), project + calibrate the pooled draws onto it and return the pooled
# selected-model posterior + summary. Reuses the standalone projection/calibration/BIC
# helpers, so the per-chain path is unchanged. Errors (a degenerate pool) are caught by
# the caller. Returns best_select / select / bic_models / loo_models /
# posterior_best_models / summary_table_selected.
.pooled_multichain <- function(X, Y, model_chains, model, standardize, grid,
                               selection_set, criterion, search, X_norm, X_mean, Y_mean) {
  D <- dim(X)[1]; n <- dim(X)[2]; p <- dim(X)[3]
  ps <- .pooled_select(X, Y, model_chains, model, standardize, grid,
                       selection_set, criterion, search)
  pc <- ps$pc; best_select <- ps$best_select; select <- ps$select
  bic_models <- ps$bic_models; loo_models <- ps$loo_models

  # ---- 5. projection + calibration onto the single pooled best model (mirrors step 13) ----
  if (standardize) {
    projection  <- projection_posterior(X, pc$post_beta, pc$post_sigma2, best_select)
    calibration <- calibrate_posterior(X, Y, projection$beta2_arr, best_select, sigma2_draws = projection$sigma2_opt)
  } else {
    projection  <- projection_posterior(X, pc$post_beta, pc$post_sigma2, best_select, alpha1_arr = pc$post_alpha)
    calibration <- calibrate_posterior(X, Y, projection$beta2_arr, best_select, sigma2_draws = projection$sigma2_opt, alpha1_arr = projection$alpha2_arr)
  }
  bc <- calibration$beta_cal_arr; bp <- projection$beta2_arr
  sel__ <- which(as.logical(best_select)); extra__ <- numeric(dim(bc)[1])
  if (length(sel__) > 0) for (d in seq_len(D)) {
    Xs__    <- X[d, , ][, sel__, drop = FALSE]
    noise__ <- matrix(bc[, d, sel__], nrow = dim(bc)[1]) - matrix(bp[, d, sel__], nrow = dim(bp)[1])
    extra__ <- extra__ + rowSums((noise__ %*% t(Xs__))^2)
  }
  extra__ <- extra__ / (n * D)
  pbm <- list(
    post_sigma2 = projection$sigma2_opt + extra__, post_beta = bc,
    post_pool_beta = matrix(bc, nrow = dim(bc)[1] * dim(bc)[2], ncol = dim(bc)[3]),
    post_alpha = projection$alpha2_arr,
    post_sigma2_projected = projection$sigma2_opt, post_beta_projected = bp,
    post_pool_beta_projected = matrix(bp, nrow = dim(bp)[1] * dim(bp)[2], ncol = dim(bp)[3]),
    post_alpha_projected = projection$alpha2_arr)

  # ---- 6. undo standardization (mirrors step 15) ----
  NPc <- dim(bc)[1]
  if (standardize) {
    pbm$post_beta_original           <- array(NA, dim = c(NPc, D, p))
    pbm$post_beta_original_projected <- array(NA, dim = c(NPc, D, p))
    for (j in 1:p) {
      pbm$post_beta_original[, , j]           <- sapply(1:D, function(d) pbm$post_beta[, d, j] / X_norm[[d]][j])
      pbm$post_beta_original_projected[, , j] <- sapply(1:D, function(d) pbm$post_beta_projected[, d, j] / X_norm[[d]][j])
    }
    pbm$post_pool_beta_original           <- matrix(pbm$post_beta_original,           NPc * D, p)
    pbm$post_pool_beta_original_projected <- matrix(pbm$post_beta_original_projected, NPc * D, p)
    pbm$post_alpha_original           <- sapply(1:D, function(d) Y_mean[[d]] - sapply(1:NPc, function(np) sum(pbm$post_beta_original[np, d, ]           * X_mean[[d]])))
    pbm$post_alpha_original_projected <- sapply(1:D, function(d) Y_mean[[d]] - sapply(1:NPc, function(np) sum(pbm$post_beta_original_projected[np, d, ] * X_mean[[d]])))
  }

  # ---- 7. pooled summary table (calibrated: pooled mean + 95% interval) ----
  bmat <- if (standardize) pbm$post_pool_beta_original else pbm$post_pool_beta
  intv <- as.numeric(if (standardize) pbm$post_alpha_original else pbm$post_alpha)
  qs <- function(m) cbind(mean = colMeans(m), q2.5 = apply(m, 2, stats::quantile, 0.025), q97.5 = apply(m, 2, stats::quantile, 0.975))
  tab <- rbind(qs(matrix(intv, ncol = 1)), qs(bmat), qs(matrix(pbm$post_sigma2, ncol = 1)))
  summ <- data.frame(variable = c("intercept", paste0("beta_", 1:p), "sigma2"), tab, row.names = NULL, check.names = FALSE)

  list(best_select = as.logical(best_select), select = select, bic_models = bic_models,
       loo_models = loo_models, posterior_best_models = pbm, summary_table_selected = summ)
}

#' Bayesian MI-LASSO for Multiply-Imputed Regression
#'
#' Fit a Bayesian multiple-imputation LASSO (BMI-LASSO) model across
#' multiply-imputed datasets, using one of four priors: Multi-Laplace,
#' Horseshoe, ARD, or Spike-Laplace. Automatically standardizes data,
#' runs MCMC in parallel, performs variable selection via four-step
#' projection predictive variable selection, and selects a final submodel by BIC.
#'
#' @param X A numeric matrix or array of predictors.  If a matrix \code{n × p},
#'   it is taken as one imputation; if an array \code{D × n × p}, each slice
#'   along the first dimension is one imputed dataset.
#' @param Y A numeric vector or matrix of outcomes.  If a vector of length \code{n},
#'   it is recycled for each imputation; if a \code{D × n} matrix, each row
#'   is the response for one imputation.
#' @param model Character; which prior to use.  One of \code{"Multi_Laplace"},
#'   \code{"Horseshoe"}, \code{"ARD"}, or \code{"Spike_Laplace"}.
#' @param standardize Logical; whether to normalize each \code{X} and centralize
#'   \code{Y} within each imputation before fitting.  Default \code{TRUE}.
#' @param search Character; how the candidate submodels along the path are
#'   generated. \code{"grid"} (default) applies a per-coefficient marginal rule
#'   over \code{grid}: for the shrinkage priors (\code{"Multi_Laplace"},
#'   \code{"Horseshoe"}, \code{"ARD"}) the symmetric credible interval that
#'   excludes 0, and for \code{"Spike_Laplace"} the posterior inclusion
#'   probability \eqn{p_j = \Pr(\gamma_j = 1 \mid y)} at or above the grid value
#'   (\code{0.5} gives the Barbieri & Berger (2004) median-probability model).
#'   \code{"snc"} uses the scaled-neighborhood criterion over \code{grid}.
#'   \code{"forward"} builds a nested path by forward stepwise search on the
#'   single-point projection loss (Piironen et al. 2020, Sec. 4).
#' @param grid Numeric vector; thresholds explored by the \code{"grid"} and
#'   \code{"snc"} searches (ignored by \code{"forward"}). Default \code{seq(0,1,0.01)}.
#' @param orthogonal Logical; if \code{TRUE}, using orthogonal approximations for
#'   degrees‐of‐freedom estimations.  Default \code{FALSE}.
#' @param nburn Integer; number of burn-in MCMC iterations per chain. Default \code{5000}.
#' @param npost Integer; number of post-burn-in samples to retain per chain. Default \code{5000}.
#' @param seed Optional integer; base random seed.  Each chain adds its index.  When
#'   supplied, BLAS is pinned to a single thread for the duration of the fit so results
#'   are bit-reproducible (multithreaded BLAS otherwise makes the MCMC draws, and hence
#'   the selection, non-deterministic).
#' @param nchains Integer; number of MCMC chains to run in parallel. Default \code{1}.
#' @param ncores Integer; number of parallel cores to use. Default \code{1}.
#' @param output_verbose Logical; print progress messages. Default \code{TRUE}.
#' @param printevery Integer; print status every so many iterations. Default \code{1000}.
#' @param selection_set Optional logical vector of length \code{p}.  If supplied, the
#'   four-step candidate-generation + BIC search is bypassed and the fitted posterior
#'   is projected and calibrated directly onto this fixed set (\code{bic_models} is then
#'   \code{NULL}).  Useful for post-selection inference under a pre-specified (e.g. the
#'   true) model.  At most \code{n - 2} variables may be selected (to keep the projection
#'   full rank and leave a residual degree of freedom for \eqn{\sigma^2}).  Default
#'   \code{NULL} (run the search).
#' @param criterion Character; model-size selection criterion along the candidate
#'   path. \code{"bic"} (default) uses the modified BIC. \code{"loo-max"} uses
#'   subject-level PSIS-LOO expected log predictive density (Piironen et al. 2020,
#'   Sec. 5.2) and selects the elpd-maximising subset. \code{"loo-1se"} applies the
#'   one-standard-error rule of Piironen et al. (2020) to the same PSIS-LOO path,
#'   selecting the smallest candidate within one standard error of the
#'   elpd-maximising one. \code{"loo"} is an alias for \code{"loo-max"}.
#' @param \dots Additional model-specific hyperparameters:
#'   - For \code{"Multi_Laplace"}: \code{h} (shape) and \code{v} (scale) of Gamma hyperprior.
#'   - For \code{"Spike_Laplace"}: \code{a} (shape) and \code{b} (scale) of Gamma hyperprior.
#'
#' @return A named list with elements:
#' \describe{
#'   \item{\code{posterior}}{List of length \code{nchains} of MCMC outputs (posterior draws).}
#'   \item{\code{select}}{List of length \code{nchains} of logical matrices showing
#'     which variables are selected at each grid value.}
#'   \item{\code{best_select}}{List of length \code{nchains} of the single best
#'     selection (by BIC) for each chain.}
#'   \item{\code{posterior_best_models}}{List of length \code{nchains} of projected
#'     posterior draws for the best submodel.}
#'   \item{\code{bic_models}}{List of length \code{nchains} of BIC values and
#'     degrees-of-freedom for each candidate submodel.}
#'   \item{\code{loo_models}}{(\code{criterion} other than \code{"bic"}) The elpd
#'     path, its pointwise contributions, the max Pareto-k per candidate, the
#'     candidate sizes, the elpd-maximising index and the one-standard-error
#'     index. \code{NULL} otherwise.}
#'   \item{\code{summary_table_full}}{A data frame summarizing rank-normalized
#'     split-Rhat and other diagnostics for the full model.}
#'   \item{\code{summary_table_selected}}{A data frame summarizing diagnostics
#'     for the selected submodel after projection.}
#'   \item{\code{pooled}}{(\code{nchains > 1} only; \code{NULL} otherwise) The
#'     multi-chain \emph{pooled} four-step selection: all chains' draws are pooled
#'     and selected ONCE, so the chains yield a single coherent submodel rather than
#'     one selection per chain.  A list with \code{best_select}, \code{select},
#'     \code{bic_models} / \code{loo_models}, \code{posterior_best_models}
#'     (calibrated + projected draws), and \code{summary_table_selected} (pooled
#'     mean and 95\% interval per coefficient).  The per-chain outputs above are
#'     unchanged.}
#' }
#'
#' @examples
#' sim <- sim_A(n = 100, p = 20, type = "MAR", SNP = 1.5, low_missing = TRUE, n_imp = 5, seed = 123)
#' X <- sim$data_MI$X
#' Y <- sim$data_MI$Y
#' fit <- BMI_LASSO(X, Y, model = "Horseshoe",
#'                  nburn = 100, npost = 100,
#'                  nchains = 1, ncores = 1)
#' str(fit$best_select)
#' @export
BMI_LASSO = function(X, Y, model, standardize = TRUE, search = "grid", grid = seq(0, 1, 0.01), orthogonal = FALSE, nburn = 5000, npost = 5000, seed = NULL, nchains = 1, ncores = 1, output_verbose = TRUE, printevery = 1000, selection_set = NULL, criterion = "bic", ...){
  # -------------------------------
  # 1. Validate input model
  # -------------------------------
  if (!model %in% c("Multi_Laplace", "Horseshoe", "ARD", "Spike_Laplace")) {
    stop("Invalid model_name. Available options: Multi_Laplace, Horseshoe, ARD, Spike_Laplace.")
  }
  if (!criterion %in% c("bic", "loo-max", "loo-1se", "loo"))
    stop('criterion must be one of "bic", "loo-max", "loo-1se".')
  if (identical(criterion, "loo")) criterion <- "loo-max"
  if (!search %in% c("grid", "snc", "forward")) stop('search must be "grid", "snc", or "forward".')


  start = Sys.time()

  # -------------------------------
  # 1b. Reproducibility guard: force single-threaded BLAS when a seed is given.
  #
  # Multithreaded BLAS (Apple Accelerate/vecLib, OpenBLAS, MKL) performs
  # non-deterministic parallel floating-point reductions in the dense matrix
  # products / Cholesky factorizations that drive every Gibbs update.  Even with
  # an identical seed, the last-bit rounding differences between runs amplify
  # through the MCMC recursion (a chaotic map) into visibly different posterior
  # draws -- and hence different candidate sets and different BIC selections.
  # When `seed` is supplied the user is asking for reproducibility, so we pin the
  # BLAS thread count to 1 for the duration of the fit and restore the caller's
  # environment on exit.  (No effect when seed = NULL: full multithreaded speed.)
  if (!is.null(seed)) {
    .blas_vars <- c("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    .old_blas  <- Sys.getenv(.blas_vars, unset = NA_character_)
    Sys.setenv(VECLIB_MAXIMUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1")
    on.exit({
      set_idx <- !is.na(.old_blas)
      if (any(set_idx))  do.call(Sys.setenv, as.list(.old_blas[set_idx]))
      if (any(!set_idx)) Sys.unsetenv(.blas_vars[!set_idx])
    }, add = TRUE)
  }

  # -------------------------------
  # 2. Reshape X into D x n x p
  # -------------------------------
  if (is.null(dim(X))) {
    X = array(X, dim = c(1, length(X), 1))
  } else if (length(dim(X)) == 2) {
    X = array(X, dim = c(1, nrow(X), ncol(X)))
  }

  D = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]

  # -------------------------------
  # 2b. Validate selection_set (if supplied): a fixed set bypasses the four-step
  #     search. Checked here, before any MCMC, so bad input fails fast.
  # -------------------------------
  if (!is.null(selection_set)) {
    if (!is.logical(selection_set) || length(selection_set) != p) {
      stop("`selection_set` must be a logical vector of length ", p,
           " (the number of predictors).")
    }
    # Same cap as the automatic BIC search (Section 11): |s| <= n - 2 keeps the
    # design [1, X_s] full rank and leaves >= 1 residual df for sigma^2; otherwise
    # the projection's solve(crossprod(X_s)) is singular / sigma^2 is undefined.
    if (sum(selection_set) > n - 2) {
      stop("`selection_set` selects ", sum(selection_set), " variables but must select at most n - 2 = ",
           n - 2, " (to keep the projection full rank and leave a residual degree of freedom for sigma^2).")
    }
  }

  # -------------------------------
  # 3. Reshape Y to D x n if needed
  # -------------------------------
  if (is.null(dim(Y))) {
    Y = t(sapply(1:D, function(i) Y))
  }
  if (dim(Y)[2] == 1) {
    Y = t(sapply(1:D, function(i) Y[, 1]))
  }

  # -------------------------------
  # 4. Save original copies for BIC
  # -------------------------------
  X_O = as.array.default(X)
  Y_O = as.matrix(Y)

  # -------------------------------
  # 5. Standardize X (if requested)
  # -------------------------------
  if (standardize == TRUE) {
    X_mean = list()
    Y_mean = list()
    X_norm = list()
    for (d in 1:D) {
      x = X[d,,]
      X_mean[[d]] = apply(x, 2, mean)
      X_norm[[d]] = apply(x, 2, stats::sd) #* sqrt(n)
      X[d,,] = scale(x)# / sqrt(n)
      Y_mean[[d]] = mean(Y[d,])
      Y[d,] = Y[d,] - Y_mean[[d]]
    }
  }

  # -------------------------------
  # 6. Handle model-specific parameters from ...
  # -------------------------------
  extra_parameters <- list(...)
  switch(model,
         "Multi_Laplace" = {
           if (!("h" %in% names(extra_parameters))) {
             if(output_verbose) cat("Missing 'h' for Multi_Laplace. Use default value h = 2", "\n")
             extra_parameters$h = 2
           }
           if (!("v" %in% names(extra_parameters))) {
             if(output_verbose) cat(paste0("Missing 'v' for Multi_Laplace. Use default value v = ", (D+1)/D / (extra_parameters$h - 1)), "\n")
             extra_parameters$v = (D+1)/D / (extra_parameters$h - 1)
           }
         },
         "Spike_Laplace" = {
           if (!("a" %in% names(extra_parameters))) {
             if(output_verbose) cat("Missing 'a' for Spike_Laplace. Use default value a = 2", "\n")
             extra_parameters$a = 2
           }
           if (!("b" %in% names(extra_parameters))) {
             if(output_verbose) cat(paste0("Missing 'b' for Spike_Laplace. Use default value b = ", (D+1)/(2 * D) / (extra_parameters$a - 1)), "\n")
             extra_parameters$b = (D+1)/(2 * D) / (extra_parameters$a - 1)
           }
         }
  )

  # -------------------------------
  # 7. Set up parallel execution
  # -------------------------------
  if (ncores == 1 || nchains == 1) {
    foreach::registerDoSEQ()
  } else {
    ## Workers are fresh R sessions: they inherit neither this session's library
    ## paths nor the package namespace, so hand both over before running tasks.
    ## The backend is reset together with the cluster, so an error part way
    ## through does not leave foreach pointing at a closed one.
    cl <- parallel::makePSOCKcluster(min(ncores, nchains))
    on.exit({ foreach::registerDoSEQ(); parallel::stopCluster(cl) }, add = TRUE)
    parallel::clusterCall(cl, function(paths) .libPaths(paths), .libPaths())
    parallel::clusterEvalQ(cl, loadNamespace("BMIselect"))
    doParallel::registerDoParallel(cl)
  }

  `%dopar%` <- foreach::`%dopar%`

  # -------------------------------
  # 8. Run MCMC chains in parallel
  # -------------------------------
  model_chains = suppressWarnings(foreach::foreach(
    chain = 1:nchains, .combine = list, .multicombine = TRUE,
    .maxcombine = ifelse(nchains >= 2, nchains, 2)
  ) %dopar% {
    seed_chain = if (!is.null(seed)) seed + chain else NULL
    return(switch(model,
                  "Multi_Laplace" = {
                    multi_laplace_mcmc(X, Y, intercept = !standardize, h = extra_parameters$h, v = extra_parameters$v,
                                       nburn = nburn, npost = npost, seed = seed_chain,
                                       verbose = output_verbose, printevery = printevery, chain_index = chain)
                  },
                  "Horseshoe" = {
                    horseshoe_mcmc(X, Y, intercept = !standardize,
                                   nburn = nburn, npost = npost, seed = seed_chain,
                                   verbose = output_verbose, printevery = printevery, chain_index = chain)
                  },
                  "ARD" = {
                    ARD_mcmc(X, Y, intercept = !standardize,
                             nburn = nburn, npost = npost, seed = seed_chain,
                             verbose = output_verbose, printevery = printevery, chain_index = chain)
                  },
                  "Spike_Laplace" = {
                    spike_laplace_partially_mcmc(X, Y, intercept = !standardize,
                                       a = extra_parameters$a, b = extra_parameters$b,
                                       nburn = nburn, npost = npost, seed = seed_chain,
                                       verbose = output_verbose, printevery = printevery, chain_index = chain)
                  }
    ))
  })


  # -------------------------------
  # 9. Handle degenerate list case
  # -------------------------------
  if ("post_sigma2" %in% names(model_chains)) {
    model_chains = list(model_chains)
  }

  # -------------------------------
  # 9b. Timestamp: MCMC (full-model) is finished here. Everything below is the
  #     four-step projection-predictive selection, so total - mcmc = four-step.
  # -------------------------------
  mcmc_time <- Sys.time()


  # -------------------------------
  # 10. Candidate generation (search = "grid" / "snc" / "forward")
  # search == "grid":
  # - For shrinkage models: use symmetric credible intervals
  # - For Spike_Laplace: threshold the posterior inclusion probability
  #   PIP_j = mean(post_gamma[, j]) (posterior mean of the inclusion
  #   indicator gamma_j); selecting {j : PIP_j >= 0.5} is the Barbieri &
  #   Berger (2004) median probability model.  NB: use post_gamma (the
  #   indicator draws), NOT post_theta (the Beta mixing weight theta_j),
  #   whose posterior mean is a shrunken surrogate and not the PIP.
  # search == "snc": scaled-neighborhood criterion.  "forward": stepwise.
  # -------------------------------

  if (is.null(selection_set)) {
  if (search == "forward") {
    select = forward_candidates(X, model_chains, n_max = n - 2, standardize = standardize)
  } else if (search == "snc") {
    select = lapply(model_chains, function(c) {
      SNCv = apply(c$post_pool_beta, 2, function(x) mean(abs(x) <= sqrt(stats::var(x))))
      se = unique(rbind(
        t(sapply(as.character(grid), function(ci) {
          ci = as.numeric(ci)
          SNCv < ci
        }, simplify = TRUE, USE.NAMES = TRUE)),
        "median" = (apply(c$post_pool_beta, 2, median) != 0)
      ))
      se
    })
  }else{
    if (model %in% c("Multi_Laplace", "Horseshoe", "ARD")) {
      select = lapply(model_chains, function(c) {
        unique(t(sapply(as.character(grid), function(ci) {
          ci = as.numeric(ci)
          apply(c$post_pool_beta, 2, function(x_j)
            prod(sign(stats::quantile((x_j), c((1 - ci)/2, 1 - (1 - ci)/2)))))
        }, simplify = TRUE, USE.NAMES = TRUE)) == 1)
      })
    } else {
        select = lapply(model_chains, function(c) {
          #print(spike_slab_threshold)
          if(length(grid)==1){
            se = unique(t(sapply(as.character(grid), function(ci) {
                ci = as.numeric(ci)
                apply(c$post_gamma, 2, function(x_j) mean(x_j) >= ci)
              }, simplify = TRUE, USE.NAMES = TRUE)))

          }else if(is.null(grid)){
            #print(123)
            se = t(as.matrix((apply(c$post_pool_beta, 2, median) != 0)))
            #print(se)
          }else{
            se = rbind(
              "median" = (apply(c$post_pool_beta, 2, median) != 0),
              unique(t(sapply(as.character(grid), function(ci) {
                ci = as.numeric(ci)
                apply(c$post_gamma, 2, function(x_j) mean(x_j) >= ci)
              }, simplify = TRUE, USE.NAMES = TRUE)))
            )
            se[,colMeans(c$post_gamma) == 1] = TRUE
            se[,colMeans(c$post_gamma) == 0] = FALSE
          }

          se
        })
      }
  }

  # -------------------------------
  # 11. Remove non-full rank selection set, AND cap |s| <= n - 2.
  #
  # The cap guards against the BIC saturation pathology that arises at p > n.
  # When |s| = n - 1 with column-centered X (and Y centered), col(X_{[s]})
  # generically spans orth(1) -- the same (n-1)-dim subspace in which Y lives
  # -- so P_{X_{[s]}} Y = Y exactly, SSE/(Dn) drops to numerical zero, and
  # BIC = log(SSE/(Dn)) + df * log(Dn)/(Dn) collapses to -infinity regardless
  # of which variables are in the subset.  Requiring |s| + 1 <= n - 1 leaves at
  # least one residual degree of freedom for sigma^2 estimation, matching the
  # classical regression requirement.  No effect when p <= n - 2.
  # -------------------------------
  n_obs <- n
  select = lapply(select, function(select_set){
    # A subset of the columns of a full-rank design is itself full rank, so if
    # [1, X_d] already has full column rank we can skip the per-candidate rank
    # check for imputation d and only fall back to it when the full design is
    # rank-deficient (p > n, or exact collinearity).  This turns the rank check
    # from O(#candidates * D) QR decompositions into O(D).
    full_ok <- sapply(1:D, function(d) qr(cbind(1, X[d,,, drop = TRUE]))$rank == p + 1)
    valid_idx <- sapply(1:nrow(select_set), function(i) {
      se <- select_set[i, ]
      sel_cols <- which(se)
      if (length(sel_cols) > n_obs - 2) return(FALSE)        # |s| <= n - 2
      if (length(sel_cols) == 0) return(TRUE)                # no variables selected
      all(sapply(1:D, function(d) {
        if (full_ok[d]) return(TRUE)                         # subset of full-rank design
        qr(cbind(1, X[d,,, drop = TRUE][, sel_cols, drop = FALSE]))$rank == (length(sel_cols) + 1)
      }))
    })
    select_set[valid_idx, , drop = FALSE]
  })

  # -------------------------------
  # 12. Model selection: BIC (default) or PSIS-LOO
  # -------------------------------
  if (criterion == "bic") {
  bic_models = foreach::foreach(
    i = seq_along(select), .combine = list, .multicombine = TRUE,
    .maxcombine = ifelse(nchains >= 2, nchains, 2)) %dopar% {
      apply(select[[i]], 1, function(subselect){
        if(standardize == TRUE)
          project_beta_sigma2 = projection_mean(X, apply(model_chains[[i]]$post_beta, c(2,3), mean), subselect, mean(model_chains[[i]]$post_sigma2))
        else
          project_beta_sigma2 = projection_mean(X, apply(model_chains[[i]]$post_beta, c(2,3), mean), subselect, mean(model_chains[[i]]$post_sigma2), colMeans(model_chains[[i]]$post_alpha))
        project_beta = project_beta_sigma2$beta2_mat
        if(sum(subselect) == 0){
          df = 0
        }else{
          switch(model,
                 "Multi_Laplace" = {
                   if(orthogonal){
                     df = D * mean(
                       apply(model_chains[[i]]$post_lambda2, 1, function(l) sum((n * l / (n * l + 1))[subselect]))
                     )
                   }else{
                     df = sum(sapply(1:D, function(d){
                       X_d = X[d,,]
                       X_ds = X_d[,subselect, drop = FALSE]
                       sum(diag(Rfast::spdinv(crossprod(X_ds)) %*% crossprod(X_ds, model_chains[[i]]$hat_matrix_proj[d,,] %*% X_ds)))
                     }))
                   }
                 },
                 "Horseshoe" = {
                   if(orthogonal){
                     df = D * mean(
                       sapply(1:npost, function(np) sum((n * model_chains[[i]]$post_lambda2[np,] * model_chains[[i]]$post_tau2[np] / (n * model_chains[[i]]$post_lambda2[np,] * model_chains[[i]]$post_tau2[np] + 1))[subselect]))
                     )
                   }else{
                     df = sum(sapply(1:D, function(d){
                       X_d = X[d,,]
                       X_ds = X_d[,subselect, drop = FALSE]
                       sum(diag(Rfast::spdinv(crossprod(X_ds)) %*% crossprod(X_ds, model_chains[[i]]$hat_matrix_proj[d,,] %*% X_ds)))
                     }))
                   }
                 },
                 "ARD" = {
                   if(orthogonal){
                     df = D * mean(
                       apply(model_chains[[i]]$post_psi2, 1, function(l) sum((n / (l + n))[subselect]))
                     )
                   }else{
                     df = sum(sapply(1:D, function(d){
                       X_d = X[d,,]
                       X_ds = X_d[,subselect, drop = FALSE]
                       sum(diag(Rfast::spdinv(crossprod(X_ds)) %*% crossprod(X_ds, model_chains[[i]]$hat_matrix_proj[d,,] %*% X_ds)))
                     }))
                   }
                 },
                 "Spike_Laplace" = {
                   if(orthogonal){
                     df = mean(sapply(1:npost, function(np) sum(D * ((n * model_chains[[i]]$post_lambda2[np,]) / (1 + n * model_chains[[i]]$post_lambda2[np,]))[model_chains[[i]]$post_Z[np,] & subselect])))
                   }else{
                     df = sum(sapply(1:D, function(d){
                       X_d = X[d,,]
                       X_ds = X_d[,subselect, drop = FALSE]
                       Xtlambda2X = model_chains[[i]]$hat_matrix_proj[d,,]
                       sum(diag(Rfast::spdinv(crossprod(X_ds)) %*% crossprod(X_ds, Xtlambda2X %*% X_ds)))
                     }))
                   }
                 }
          )
        }
        if(standardize == TRUE)
          return(c("BIC" = compute_mi_bic(X, Y, project_beta, df = df), "df" = df))
        else
          return(c("BIC" = compute_mi_bic(X, Y, project_beta, df = df, alpha = project_beta_sigma2$alpha2_vec), "df" = df))
      })
    }

  if(any(class(bic_models) != "list")) bic_models = list(bic_models)
  best_select <- sapply(seq_along(bic_models), function(i) {
    select[[i]][which.min(bic_models[[i]][1,]), , drop = FALSE]
  }, simplify = FALSE)
  loo_models <- NULL
  } else {
    loo_models  <- loo_select(X, Y, model_chains, select, standardize = standardize)
    best_select <- lapply(seq_along(loo_models), function(i) {
      idx <- if (criterion == "loo-1se") loo_models[[i]]$best_1se else loo_models[[i]]$best
      select[[i]][idx, , drop = FALSE]
    })
    bic_models  <- NULL
  }
  } else {
    # selection_set supplied: bypass the four-step search (candidate + rank + BIC) and
    # project/calibrate directly onto the fixed set. bic_models is NULL.
    select      <- NULL
    bic_models  <- NULL
    loo_models  <- NULL
    best_select <- lapply(seq_along(model_chains),
                          function(i) matrix(as.logical(selection_set), nrow = 1))
  }

  # -------------------------------
  # 12b. Multi-chain: replace the per-chain selections with ONE selection made on the
  #      pooled draws and project EVERY chain onto it. Otherwise each chain runs its
  #      own four-step and the chains can select different variables, so the
  #      cross-chain split-Rhat of the selected model conflates that selection
  #      disagreement with genuine non-convergence of the sampler. Projecting all
  #      chains onto the shared submodel makes the selected-model Rhat a clean
  #      convergence diagnostic. (A single chain, or a user-fixed selection_set,
  #      already shares one selection, so both are left untouched.)
  # -------------------------------
  if (nchains > 1 && is.null(selection_set)) {
    shared      <- .pooled_select(X, Y, model_chains, model, standardize, grid,
                                  selection_set, criterion, search)
    best_select <- rep(list(shared$best_select), nchains)
    select      <- rep(list(shared$select),      nchains)
    if (criterion == "bic") bic_models <- rep(list(shared$bic_models), nchains)
    else                    loo_models <- rep(list(shared$loo_models), nchains)
  }

  # -------------------------------
  # 13. Project on the selected posterior distribution
  # -------------------------------
  posterior_best_models = foreach::foreach(
    ij = seq_along(best_select), .combine = list, .multicombine = TRUE,
    .maxcombine = ifelse(nchains >= 2, nchains, 2)) %dopar% {

    if(standardize == TRUE) {
      projection  = projection_posterior(X, model_chains[[ij]]$post_beta, model_chains[[ij]]$post_sigma2, best_select[[ij]])
      calibration = calibrate_posterior(X, Y, projection$beta2_arr, best_select[[ij]], sigma2_draws = projection$sigma2_opt)
    } else {
      projection  = projection_posterior(X, model_chains[[ij]]$post_beta, model_chains[[ij]]$post_sigma2, best_select[[ij]], alpha1_arr = model_chains[[ij]]$post_alpha)
      calibration = calibrate_posterior(X, Y, projection$beta2_arr, best_select[[ij]], sigma2_draws = projection$sigma2_opt, alpha1_arr = projection$alpha2_arr)
    }
    bc <- calibration$beta_cal_arr    # calibrated draws
    bp <- projection$beta2_arr        # pushforward (projected) draws

    # sigma^2 recomputed from the CALIBRATED beta (whole-model consistency): the ONE
    # shared sigma^2 (df ~ D(n+p) = Dn, UNCHANGED) plus the projection loss evaluated
    # at the *calibrated* beta_s (not the projected beta_s). The loss uses the FULL-
    # MODEL fitted values (projpred-consistent). By orthogonality of the projection
    # residual, the extra over the projected sigma_s^2 is
    #   (1/nD) sum_d || X_s^d (beta_cal - beta_proj)^(t,d) ||^2,
    # i.e. the deficit noise propagated into the residual variance -> a coherent
    # sigma_s^2 that moves with the beta calibration. Under-coverage of the complete-
    # data sigma^2 is the projpred sigma-absorption (removed variables), by design.
    sel__ <- which(as.logical(best_select[[ij]])); nn__ <- dim(X)[2]; DD__ <- dim(X)[1]
    extra__ <- numeric(dim(bc)[1])
    if (length(sel__) > 0) for (d in seq_len(DD__)) {
      Xs__    <- X[d, , ][, sel__, drop = FALSE]
      noise__ <- matrix(bc[, d, sel__], nrow = dim(bc)[1]) - matrix(bp[, d, sel__], nrow = dim(bp)[1])
      extra__ <- extra__ + rowSums((noise__ %*% t(Xs__))^2)
    }
    extra__ <- extra__ / (nn__ * DD__)

    return(list(
      # ---- CALIBRATED (primary output; drives summary_table_selected) ----
      post_sigma2     = projection$sigma2_opt + extra__,
      post_beta       = bc,
      post_pool_beta  = matrix(bc, nrow = dim(bc)[1] * dim(bc)[2], ncol = dim(bc)[3]),
      post_alpha      = projection$alpha2_arr,
      post_pool_alpha = as.numeric(projection$alpha2_arr),
      # ---- PUSHFORWARD projection (kept for comparison; drop at release) ----
      post_sigma2_projected     = projection$sigma2_opt,
      post_beta_projected       = bp,
      post_pool_beta_projected  = matrix(bp, nrow = dim(bp)[1] * dim(bp)[2], ncol = dim(bp)[3]),
      post_alpha_projected      = projection$alpha2_arr,
      post_pool_alpha_projected = as.numeric(projection$alpha2_arr)
    ))
    }
  if(length(posterior_best_models) != nchains) posterior_best_models = list(posterior_best_models)




  # -------------------------------
  # 14. Reset parallel backend
  # -------------------------------
  if (ncores > 1) {
    doParallel::stopImplicitCluster()
    foreach::registerDoSEQ()
  }


  # -------------------------------
  # 15. Undo standardization (if requested)
  # Convert posterior draws back to original scale
  # -------------------------------

  # summary helper: full posterior summary incl. rank-normalized split-Rhat / ESS.
  .summ <- function(rv) posterior::summarize_draws(rv)

  if (standardize == TRUE) {
    for (chain in 1:nchains) {
      model_chains[[chain]][["post_beta_original"]] = array(NA, dim = c(npost, D, p))
      model_chains[[chain]][["post_pool_beta_original"]] = matrix(NA, nrow = npost * D, ncol = p)
      posterior_best_models[[chain]][["post_beta_original"]] = array(NA, dim = c(npost, D, p))
      posterior_best_models[[chain]][["post_pool_beta_original"]] = matrix(NA, nrow = npost * D, ncol = p)
      posterior_best_models[[chain]][["post_beta_original_projected"]] = array(NA, dim = c(npost, D, p))
      posterior_best_models[[chain]][["post_pool_beta_original_projected"]] = matrix(NA, nrow = npost * D, ncol = p)
      for (j in 1:p) {
        model_chains[[chain]][["post_beta_original"]][,,j] =
          sapply(1:D, function(d) model_chains[[chain]][["post_beta"]][,d,j] / X_norm[[d]][j])
        model_chains[[chain]][["post_pool_beta_original"]][,j] =
          model_chains[[chain]][["post_beta_original"]][,,j]


        posterior_best_models[[chain]][["post_beta_original"]][,,j] =
          sapply(1:D, function(d) posterior_best_models[[chain]][["post_beta"]][,d,j] / X_norm[[d]][j])
        posterior_best_models[[chain]][["post_pool_beta_original"]][,j] =
          posterior_best_models[[chain]][["post_beta_original"]][,,j]

        posterior_best_models[[chain]][["post_beta_original_projected"]][,,j] =
          sapply(1:D, function(d) posterior_best_models[[chain]][["post_beta_projected"]][,d,j] / X_norm[[d]][j])
        posterior_best_models[[chain]][["post_pool_beta_original_projected"]][,j] =
          posterior_best_models[[chain]][["post_beta_original_projected"]][,,j]
      }

      model_chains[[chain]][["post_alpha_original"]] = sapply(1:D, function(d) Y_mean[[d]] - sapply(1:npost, function(np) sum(model_chains[[chain]][["post_beta_original"]][np,d,] * X_mean[[d]])))
      posterior_best_models[[chain]][["post_alpha_original"]] = sapply(1:D, function(d) Y_mean[[d]] - sapply(1:npost, function(np) sum(posterior_best_models[[chain]][["post_beta_original"]][np,d,] * X_mean[[d]])))
      posterior_best_models[[chain]][["post_alpha_original_projected"]] = sapply(1:D, function(d) Y_mean[[d]] - sapply(1:npost, function(np) sum(posterior_best_models[[chain]][["post_beta_original_projected"]][np,d,] * X_mean[[d]])))
    }

    rvar_beta_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_pool_beta_original), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_sigma2 = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_intercept_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) as.numeric(chain$post_alpha_original)), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_full = rbind(
      .summ(rvar_intercept_pool),
      .summ(rvar_beta_pool),
      .summ(rvar_sigma2)
    )
    summary_table_full$variable = stringr::str_remove(summary_table_full$variable, "rvar_")

    select_rvar_beta_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_beta_original), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_sigma2 = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_intercept_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) as.numeric(chain$post_alpha_original)), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_select = rbind(
      .summ(select_rvar_intercept_pool),
      .summ(select_rvar_beta_pool),
      .summ(select_rvar_sigma2)
    )
    summary_table_select$variable = stringr::str_remove(summary_table_select$variable, "select_rvar_")

    # pushforward (projected) comparison table (drop at release)
    select_rvar_beta_pool_projected = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_beta_original_projected), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_sigma2_projected = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_sigma2_projected), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_intercept_pool_projected = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) as.numeric(chain$post_alpha_original_projected)), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_select_projected = rbind(
      .summ(select_rvar_intercept_pool_projected),
      .summ(select_rvar_beta_pool_projected),
      .summ(select_rvar_sigma2_projected)
    )
    summary_table_select_projected$variable = stringr::str_remove(summary_table_select_projected$variable, "select_rvar_")
    summary_table_select_projected$variable = stringr::str_remove(summary_table_select_projected$variable, "_projected")
  }else{
    rvar_beta_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_pool_beta), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_sigma2 = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_intercept_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) as.numeric(chain$post_alpha)), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_full = rbind(
      .summ(rvar_intercept_pool),
      .summ(rvar_beta_pool),
      .summ(rvar_sigma2)
    )
    summary_table_full$variable = stringr::str_remove(summary_table_full$variable, "rvar_")

    select_rvar_beta_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_beta), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_sigma2 = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_intercept_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_alpha), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_select = rbind(
      .summ(select_rvar_intercept_pool),
      .summ(select_rvar_beta_pool),
      .summ(select_rvar_sigma2)
    )
    summary_table_select$variable = stringr::str_remove(summary_table_select$variable, "select_rvar_")

    # pushforward (projected) comparison table (drop at release)
    select_rvar_beta_pool_projected = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_beta_projected), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_sigma2_projected = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_sigma2_projected), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_intercept_pool_projected = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_alpha_projected), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_select_projected = rbind(
      .summ(select_rvar_intercept_pool_projected),
      .summ(select_rvar_beta_pool_projected),
      .summ(select_rvar_sigma2_projected)
    )
    summary_table_select_projected$variable = stringr::str_remove(summary_table_select_projected$variable, "select_rvar_")
    summary_table_select_projected$variable = stringr::str_remove(summary_table_select_projected$variable, "_projected")
  }

  # -------------------------------
  # 16. Convergence diagnostics (pooled, imputations interleaved) + warning
  #
  # summarize_draws() above computed rhat/ess on the imputation-BLOCKED pooled
  # draws, which over-states non-convergence (see .rhat_ess_pooled).  We overwrite
  # ONLY the rhat / ess_bulk / ess_tail columns with the interleaved-pooled values
  # (one rank-normalized split-Rhat per coefficient, across all chains); the pooled
  # mean/median/sd/quantiles (the reported inference) are left untouched.  Row order
  # matches the summary tables:
  # intercept (1 row), beta_1..p (p rows), sigma2 (1 row).
  # -------------------------------
  {  # convergence diagnostics (always computed)
  .as3d <- function(m) array(m, dim = c(nrow(m), ncol(m), 1L))  # [np,D] -> [np,D,1]

  # Two sets of selected-model draws: PROJECTED (pushforward = actual MCMC draws)
  # and CALIBRATED (projected + i.i.d. deficit noise). Each summary table gets the
  # MATCHING diagnostics (calibrated table <- calibrated, projected table <-
  # projected). The convergence WARNING (below) uses the PROJECTED R-hat, because
  # the calibration noise inflates ESS and pulls R-hat toward 1, masking non-
  # convergence -- so it must not gate the alarm.
  if (standardize == TRUE) {
    beta_full <- lapply(model_chains,        function(ch) ch$post_beta_original)
    int_full  <- lapply(model_chains,        function(ch) .as3d(ch$post_alpha_original))
    beta_sel  <- lapply(posterior_best_models, function(ch) ch$post_beta_original_projected)
    int_sel   <- lapply(posterior_best_models, function(ch) .as3d(ch$post_alpha_original_projected))
  } else {
    beta_full <- lapply(model_chains,        function(ch) ch$post_beta)
    int_full  <- lapply(model_chains,        function(ch) .as3d(ch$post_alpha))
    beta_sel  <- lapply(posterior_best_models, function(ch) ch$post_beta_projected)
    int_sel   <- lapply(posterior_best_models, function(ch) .as3d(ch$post_alpha_projected))
  }
  sig_full <- lapply(model_chains,        function(ch) array(ch$post_sigma2, dim = c(length(ch$post_sigma2), 1L, 1L)))
  sig_sel  <- lapply(posterior_best_models, function(ch) array(ch$post_sigma2_projected, dim = c(length(ch$post_sigma2_projected), 1L, 1L)))

  dgi_f <- .rhat_ess_pooled(int_full);  dgb_f <- .rhat_ess_pooled(beta_full);  dgs_f <- .rhat_ess_pooled(sig_full)
  dgi_s <- .rhat_ess_pooled(int_sel);   dgb_s <- .rhat_ess_pooled(beta_sel);   dgs_s <- .rhat_ess_pooled(sig_sel)

  # calibrated selected-model draws -> informational R-hat/ESS. NOTE: calibration
  # adds i.i.d. deficit noise per draw, which inflates ESS and pulls R-hat toward
  # 1, so this is OPTIMISTIC; the projected R-hat above is the true convergence
  # check. Reported for transparency since the inference columns are calibrated.
  if (standardize == TRUE) {
    beta_sel_cal <- lapply(posterior_best_models, function(ch) ch$post_beta_original)
    int_sel_cal  <- lapply(posterior_best_models, function(ch) .as3d(ch$post_alpha_original))
  } else {
    beta_sel_cal <- lapply(posterior_best_models, function(ch) ch$post_beta)
    int_sel_cal  <- lapply(posterior_best_models, function(ch) .as3d(ch$post_alpha))
  }
  sig_sel_cal <- lapply(posterior_best_models, function(ch) array(ch$post_sigma2, dim = c(length(ch$post_sigma2), 1L, 1L)))
  dgi_sc <- .rhat_ess_pooled(int_sel_cal); dgb_sc <- .rhat_ess_pooled(beta_sel_cal); dgs_sc <- .rhat_ess_pooled(sig_sel_cal)

  fix_tbl <- function(tbl, di, db, ds) {
    rh <- c(di$rhat, db$rhat, ds$rhat)
    eb <- c(di$ess_bulk, db$ess_bulk, ds$ess_bulk)
    et <- c(di$ess_tail, db$ess_tail, ds$ess_tail)
    if (nrow(tbl) == length(rh)) {
      if ("rhat"     %in% names(tbl)) tbl$rhat     <- rh
      if ("ess_bulk" %in% names(tbl)) tbl$ess_bulk <- eb
      if ("ess_tail" %in% names(tbl)) tbl$ess_tail <- et
    }
    tbl
  }
  summary_table_full             <- fix_tbl(summary_table_full,             dgi_f,  dgb_f,  dgs_f)
  summary_table_select           <- fix_tbl(summary_table_select,           dgi_sc, dgb_sc, dgs_sc)  # calibrated (primary)
  summary_table_select_projected <- fix_tbl(summary_table_select_projected, dgi_s,  dgb_s,  dgs_s)   # projected

  rhat_full_max <- max(c(dgi_f$rhat, dgb_f$rhat, dgs_f$rhat), na.rm = TRUE)
  rhat_sel_max  <- max(c(dgi_s$rhat, dgb_s$rhat, dgs_s$rhat), na.rm = TRUE)
  rhat_sel_cal_max <- max(c(dgi_sc$rhat, dgb_sc$rhat, dgs_sc$rhat), na.rm = TRUE)

  # Always report the three max R-hats (full / selected calibrated / selected
  # projected). summary_table_selected is calibrated, so its primary "selected
  # model" line is the calibrated one.
  if (output_verbose) {
    cat(sprintf("Max rank-normalized split-Rhat (full model) = %.3f\n", rhat_full_max))
    cat(sprintf("Max rank-normalized split-Rhat (selected model) = %.3f\n", rhat_sel_cal_max))
    cat(sprintf("Max rank-normalized split-Rhat (selected model, projected) = %.3f\n", rhat_sel_max))
  }
  # Convergence warnings judged on the ACTUAL-MCMC draws (full model, and the
  # PROJECTED selected draws): the calibrated R-hat is masked by the deficit noise
  # and must not gate the alarm. Threshold 1.01 is the rank-normalized split-Rhat
  # convergence standard (Vehtari, Gelman, Simpson, Carpenter & Buerkner 2021).
  if (is.finite(rhat_full_max) && rhat_full_max > 1.01) {
    warning(sprintf("Full model doesn't converge. Please increase burn-in or posterior samples. The maximum rank-normalized split-Rhat is %.3f", rhat_full_max))
  }
  if (is.finite(rhat_sel_max) && rhat_sel_max > 1.01) {
    warning(sprintf("Selected model (projected draws) may not have converged. Please increase burn-in or posterior samples. The maximum rank-normalized split-Rhat is %.3f. All chains are projected onto the same pooled submodel, so this reflects sampler convergence on that submodel rather than disagreement about which variables were selected.", rhat_sel_max))
  }
  }  # end convergence diagnostics




  # -------------------------------
  # 16c. Additive multi-chain pooled selection (nchains > 1 only; pooled = NULL
  #      otherwise). Pools all chains' draws and runs the four-step selection ONCE
  #      so the chains give ONE coherent selected model. Leaves every per-chain
  #      output untouched; wrapped in tryCatch so a degenerate pool never breaks
  #      the main result.
  # -------------------------------
  pooled <- NULL
  if (nchains > 1) {
    pooled <- tryCatch(
      .pooled_multichain(X, Y, model_chains, model, standardize, grid,
                         selection_set, criterion, search,
                         if (standardize) X_norm else NULL,
                         if (standardize) X_mean else NULL,
                         if (standardize) Y_mean else NULL),
      error = function(e) { warning("pooled multi-chain selection failed: ", conditionMessage(e)); NULL })
  }

  # -------------------------------
  # 17. Handle single-chain case
  # -------------------------------
  if (length(model_chains) == 1) {
    if (!is.null(select))     select <- select[[1]]
    model_chains <- model_chains[[1]]
    if (!is.null(bic_models)) bic_models <- bic_models[[1]]
    if (!is.null(loo_models)) loo_models <- loo_models[[1]]
    posterior_best_models = posterior_best_models[[1]]
    best_select <- best_select[[1]]
  }

  # -------------------------------
  # 18. Report timing (decompose: total = full-model MCMC + four-step selection)
  # -------------------------------
  end <- Sys.time()
  total_time     <- as.numeric(difftime(end,       start,     units = "mins"))
  mcmc_only_time <- as.numeric(difftime(mcmc_time, start,     units = "mins"))  # full-model MCMC
  selection_time <- as.numeric(difftime(end,       mcmc_time, units = "mins"))  # four-step = total - MCMC
  if (output_verbose) {
    cat(sprintf("Running time for %d %s: %.2f min total (full-model MCMC %.2f + four-step selection %.2f)\n",
                nchains, ifelse(nchains > 1, "chains", "chain"),
                total_time, mcmc_only_time, selection_time))
  }

  return(list(posterior = model_chains, select = select, best_select = best_select, posterior_best_models = posterior_best_models, bic_models = bic_models, loo_models = loo_models, summary_table_full = summary_table_full, summary_table_selected = summary_table_select, summary_table_selected_projected = summary_table_select_projected,
              runtime = c(total = total_time, mcmc = mcmc_only_time, selection = selection_time),
              pooled = pooled))
}








