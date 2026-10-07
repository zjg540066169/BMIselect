## ===========================================================================
## select_criteria.R
##
## Alternative *candidate generation* (forward stepwise search) and
## *model-size selection criterion* (PSIS-LOO) for the four-step projection
## predictive procedure. Both operate on the ALREADY-FITTED full-model
## posterior draws in `model_chains`, reusing projection_mean() /
## projection_posterior(); no model is refitted. The default behaviour of
## BMI_LASSO (SNC candidate generation + modified BIC) is unchanged.
##
## References: Piironen, Paasiniemi & Vehtari (2020), "Projective inference in
## high-dimensional problems", EJS 14(1):2155-2197 -- Sec. 4 (forward search on
## the single-point projection) and Sec. 5.2 (PSIS-LOO).
## ===========================================================================

## ---------------------------------------------------------------------------
## Forward-stepwise candidate path (Piironen et al. 2020, Sec. 4).
##
## Starting from the empty set, at each step add the single variable that most
## reduces the summed single-point projection loss
##     L(s) = sum_d || X^d beta_bar^d - X^d beta_s^d ||_2^2 ,
## where beta_bar is the full-model posterior mean and beta_s^d its single-point
## (mean) projection onto the current subset s (projection_mean()). Per Sec. 4
## the penalisation/search is used ONLY to order the variables; the submodels
## along the path are fitted later by the unpenalised draw-by-draw projection.
##
## Returns, per chain, a logical matrix whose rows are the nested subsets
## s_1 subset s_2 subset ... (one variable added per row), matching the SNC
## `select` format so the downstream filter / criterion code is shared.
## ---------------------------------------------------------------------------
forward_candidates <- function(X, model_chains, n_max, standardize = TRUE) {
  D <- dim(X)[1]; n <- dim(X)[2]; p <- dim(X)[3]
  n_max <- max(1L, min(p, as.integer(n_max)))
  lapply(model_chains, function(cc) {
    beta_bar  <- apply(cc$post_beta, c(2, 3), mean)               # D x p
    alpha_bar <- if (!standardize) colMeans(cc$post_alpha) else NULL
    ## reference fitted values per imputation
    ref_fit <- lapply(seq_len(D), function(d) {
      f <- as.numeric(X[d, , ] %*% beta_bar[d, ])
      if (!is.null(alpha_bar)) f <- f + alpha_bar[d]
      f
    })
    active    <- logical(p)
    remaining <- seq_len(p)
    rows      <- vector("list", n_max)
    for (step in seq_len(n_max)) {
      losses <- vapply(remaining, function(j) {
        s <- active; s[j] <- TRUE
        pm <- projection_mean(X, beta_bar, s, 0, alpha1_vec = alpha_bar)
        sum(vapply(seq_len(D), function(d) {
          proj_fit <- as.numeric(X[d, , ] %*% pm$beta2_mat[d, ])
          if (!is.null(alpha_bar)) proj_fit <- proj_fit + pm$alpha2_vec[d]
          sum((ref_fit[[d]] - proj_fit)^2)
        }, numeric(1)))
      }, numeric(1))
      j_best         <- remaining[which.min(losses)]
      active[j_best] <- TRUE
      remaining      <- remaining[remaining != j_best]
      rows[[step]]   <- active
      if (length(remaining) == 0) break
    }
    do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
  })
}

## ---------------------------------------------------------------------------
## Per-subject leave-one-out log-likelihood matrix (npost x n).
##
## A subject i contributes ONE LOO point whose per-draw log-likelihood is summed
## over the D imputations, i.e. leave-one-subject-out across all imputations:
##     ll[t, i] = sum_d log N( Y[d,i] ; x_{i,s}^d beta_s^{d,(t)}, sigma_s^{2,(t)} ).
## Feeding this to loo::loo() gives PSIS weights w_i^(t) ~ 1 / prod_d p(y_i | .),
## the correct subject-level LOO weight for the grouped likelihood.
## sigma2_opt = full-model sigma^2 + projection loss / (nD) >= full sigma^2 > 0,
## so the normal density is always well defined.
## ---------------------------------------------------------------------------
.loglik_subject <- function(X, Y, proj, s_idx, standardize) {
  npost <- dim(proj$beta2_arr)[1]; D <- dim(X)[1]; n <- dim(X)[2]
  s2      <- proj$sigma2_opt                        # length npost, > 0
  lognorm <- -0.5 * log(2 * pi) - 0.5 * log(s2)     # length npost
  ll <- matrix(0, npost, n)
  for (d in seq_len(D)) {
    if (length(s_idx) > 0) {
      Bd  <- matrix(proj$beta2_arr[, d, s_idx], npost, length(s_idx))  # npost x |s|
      Xds <- X[d, , ][, s_idx, drop = FALSE]                          # n x |s|
      fit <- Bd %*% t(Xds)                                            # npost x n
    } else {
      fit <- matrix(0, npost, n)
    }
    if (!standardize) fit <- fit + proj$alpha2_arr[, d]               # + intercept (per draw)
    resid <- fit - matrix(Y[d, ], npost, n, byrow = TRUE)            # npost x n
    ll    <- ll + (lognorm - (resid^2) / (2 * s2))
  }
  ll
}

## ---------------------------------------------------------------------------
## PSIS-LOO model-size selection along a candidate path (Piironen et al. 2020,
## Sec. 5.2). For each candidate the full posterior is projected (draw-by-draw)
## and the subject-level LOO expected log predictive density is estimated with
## Pareto-smoothed importance sampling (loo package). Returns, per chain, the
## elpd path, the max Pareto-k diagnostic per candidate (k <= 0.7 => reliable),
## and the index of the elpd-maximising subset and of the one-standard-error subset.
## ---------------------------------------------------------------------------
## One-standard-error rule (Piironen et al. 2020, Eq. (22)-(23)): among the
## candidates whose mean pointwise elpd difference from the elpd-maximising
## candidate, plus the standard error of that paired difference, is non-negative,
## take the one with the fewest variables.
.pick_1se <- function(elpd, elpd_pw, sizes) {
  ub <- elpd_pw[[which.max(elpd)]]
  ok <- vapply(seq_along(elpd), function(k) {
    dd <- elpd_pw[[k]] - ub
    mean(dd) + stats::sd(dd) / sqrt(length(dd)) >= 0
  }, logical(1))
  oi <- which(ok)
  oi[which.min(sizes[oi])]
}

loo_select <- function(X, Y, model_chains, select, standardize = TRUE) {
  lapply(seq_along(model_chains), function(i) {
    cc   <- model_chains[[i]]
    cand <- select[[i]]
    if (is.null(dim(cand))) cand <- matrix(cand, nrow = 1)
    alpha1 <- if (!standardize) cc$post_alpha else NULL
    K    <- nrow(cand)
    elpd <- rep(NA_real_, K)
    pk   <- rep(NA_real_, K)
    pw   <- vector("list", K)                 # pointwise elpd per candidate (for 1-SE rule)
    for (r in seq_len(K)) {
      s    <- as.logical(cand[r, ])
      proj <- projection_posterior(X, cc$post_beta, cc$post_sigma2, s, alpha1_arr = alpha1)
      ll   <- .loglik_subject(X, Y, proj, which(s), standardize)
      lr   <- loo::loo(ll, r_eff = rep(1, ncol(ll)))
      elpd[r] <- lr$estimates["elpd_loo", "Estimate"]
      pk[r]   <- suppressWarnings(max(lr$diagnostics$pareto_k, na.rm = TRUE))
      pw[[r]] <- lr$pointwise[, "elpd_loo"]
    }
    sizes <- rowSums(cand)
    list(elpd = elpd, elpd_pw = pw, pareto_k = pk, size = sizes,
         best = which.max(elpd), best_1se = .pick_1se(elpd, pw, sizes))
  })
}
