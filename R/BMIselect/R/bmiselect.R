if (getRversion() >= "2.15.1") {
  utils::globalVariables(c("i", "ij"))
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
#' @param SNC Logical; if \code{TRUE}, use scaled neighborhood criterion;
#'   otherwise apply thresholding or median‐based selection. Default \code{TRUE}.
#' @param grid Numeric vector; grid of scaled neighborhood criterion (or thresholding) to explore.
#'   Default \code{seq(0,1,0.01)}.
#' @param orthogonal Logical; if \code{TRUE}, using orthogonal approximations for
#'   degrees‐of‐freedom estimations.  Default \code{FALSE}.
#' @param nburn Integer; number of burn-in MCMC iterations per chain. Default \code{4000}.
#' @param npost Integer; number of post-burn-in samples to retain per chain. Default \code{4000}.
#' @param seed Optional integer; base random seed.  Each chain adds its index.
#' @param nchains Integer; number of MCMC chains to run in parallel. Default \code{1}.
#' @param ncores Integer; number of parallel cores to use. Default \code{1}.
#' @param output_verbose Logical; print progress messages. Default \code{TRUE}.
#' @param printevery Integer; print status every so many iterations. Default \code{1000}.
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
#'   \item{\code{summary_table_full}}{A data frame summarizing rank-normalized
#'     split-Rhat and other diagnostics for the full model.}
#'   \item{\code{summary_table_selected}}{A data frame summarizing diagnostics
#'     for the selected submodel after projection.}
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
BMI_LASSO = function(X, Y, model, standardize = TRUE, SNC = TRUE, grid = seq(0, 1, 0.01), orthogonal = FALSE, nburn = 4000, npost = 4000, seed = NULL, nchains = 1, ncores = 1, output_verbose = TRUE, printevery = 1000, ...){
  # -------------------------------
  # 1. Validate input model
  # -------------------------------
  if (!model %in% c("Multi_Laplace", "Horseshoe", "ARD", "Spike_Laplace")) {
    stop("Invalid model_name. Available options: Multi_Laplace, Horseshoe, ARD, Spike_Laplace.")
  }


  start = Sys.time()

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
  if (ncores == 1) {
    foreach::registerDoSEQ()
  } else {
    doParallel::registerDoParallel(ncores)
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
  # 10. Scaled Neighborhood Criterion
  # If SNC is FALSE:
  # - For shrinkage models: use symmetric credible intervals
  # - For Spike_Laplace: use posterior median threshold
  # -------------------------------

  if(SNC){
    select = lapply(model_chains, function(c) {
      SNC = apply(c$post_pool_beta, 2, function(x) mean(abs(x) <= sqrt(stats::var(x))))
      se = unique(rbind(
        t(sapply(as.character(grid), function(ci) {
          ci = as.numeric(ci)
          SNC < ci
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
                apply(c$post_theta, 2, function(x_j) mean(x_j) >= ci)
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
                apply(c$post_theta, 2, function(x_j) mean(x_j) >= ci)
              }, simplify = TRUE, USE.NAMES = TRUE)))
            )
            se[,colMeans(c$post_theta) == 1] = TRUE
            se[,colMeans(c$post_theta) == 0] = FALSE
          }

          se
        })
      }
  }

  # -------------------------------
  # 11. Remove non-full rank selection set
  # -------------------------------
  select = lapply(select, function(select_set){
    valid_idx <- sapply(1:nrow(select_set), function(i) {
      se <- select_set[i, ]
      all(sapply(1:D, function(d) {
        sel_cols <- which(se)
        if (length(sel_cols) == 0) return(TRUE)  # no variables selected, skip rank check
        X_sub <- cbind(1, X[d,,, drop = TRUE][, sel_cols, drop = FALSE])
        qr(X_sub)$rank == (length(sel_cols) + 1)
      }))
    })

    valid_sets <- select_set[valid_idx, , drop = FALSE]
    valid_sets
  })

  # -------------------------------
  # 12. BIC calculation
  # -------------------------------
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
                       sum(diag(X_ds %*% Rfast::spdinv(t(X_ds) %*% X_ds) %*% t(X_ds) %*% model_chains[[i]]$hat_matrix_proj[d,,]))
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
                       sum(diag(X_ds %*% Rfast::spdinv(t(X_ds) %*% X_ds) %*% t(X_ds) %*% model_chains[[i]]$hat_matrix_proj[d,,]))
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
                       sum(diag(X_ds %*% Rfast::spdinv(t(X_ds) %*% X_ds) %*% t(X_ds) %*% model_chains[[i]]$hat_matrix_proj[d,,]))
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
                       sum(diag(X_ds %*% Rfast::spdinv(t(X_ds) %*% X_ds) %*% t(X_ds) %*% Xtlambda2X))
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

  # -------------------------------
  # 13. Project on the selected posterior distribution
  # -------------------------------
  posterior_best_models = foreach::foreach(
    ij = seq_along(best_select), .combine = list, .multicombine = TRUE,
    .maxcombine = ifelse(nchains >= 2, nchains, 2)) %dopar% {

    if(standardize == TRUE)
      projection =  projection_posterior(X, model_chains[[ij]]$post_beta, model_chains[[ij]]$post_sigma2, best_select[[ij]])
    else
      projection =  projection_posterior(X, model_chains[[ij]]$post_beta, model_chains[[ij]]$post_sigma2, best_select[[ij]], alpha1_arr = model_chains[[ij]]$post_alpha)


    return(list(
      post_sigma2 = projection$sigma2_opt,
      post_beta = projection$beta2_arr,
      post_pool_beta = matrix(projection$beta2_arr, nrow = dim(projection$beta2_arr)[1] * dim(projection$beta2_arr)[2],
                              ncol = dim(projection$beta2_arr)[3]),
      post_alpha = projection$alpha2_arr,
      post_pool_alpha = as.numeric(projection$alpha2_arr)
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
  if (standardize == TRUE) {
    for (chain in 1:nchains) {
      model_chains[[chain]][["post_beta_original"]] = array(NA, dim = c(npost, D, p))
      model_chains[[chain]][["post_pool_beta_original"]] = matrix(NA, nrow = npost * D, ncol = p)
      posterior_best_models[[chain]][["post_beta_original"]] = array(NA, dim = c(npost, D, p))
      posterior_best_models[[chain]][["post_pool_beta_original"]] = matrix(NA, nrow = npost * D, ncol = p)
      for (j in 1:p) {
        model_chains[[chain]][["post_beta_original"]][,,j] =
          sapply(1:D, function(d) model_chains[[chain]][["post_beta"]][,d,j] / X_norm[[d]][j])
        model_chains[[chain]][["post_pool_beta_original"]][,j] =
          model_chains[[chain]][["post_beta_original"]][,,j]


        posterior_best_models[[chain]][["post_beta_original"]][,,j] =
          sapply(1:D, function(d) posterior_best_models[[chain]][["post_beta"]][,d,j] / X_norm[[d]][j])
        posterior_best_models[[chain]][["post_pool_beta_original"]][,j] =
          posterior_best_models[[chain]][["post_beta_original"]][,,j]
      }

      model_chains[[chain]][["post_alpha_original"]] = sapply(1:D, function(d) Y_mean[[d]] - sapply(1:npost, function(np) sum(model_chains[[chain]][["post_beta_original"]][np,d,] * X_mean[[d]] / X_norm[[d]])))
      posterior_best_models[[chain]][["post_alpha_original"]] = sapply(1:D, function(d) Y_mean[[d]] - sapply(1:npost, function(np) sum(posterior_best_models[[chain]][["post_beta_original"]][np,d,] * X_mean[[d]] / X_norm[[d]])))
    }

    rvar_beta_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_pool_beta_original), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_sigma2 = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_intercept_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) as.numeric(chain$post_alpha_original)), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_full = rbind(
      posterior::summarize_draws(rvar_intercept_pool),
      posterior::summarize_draws(rvar_beta_pool),
      posterior::summarize_draws(rvar_sigma2)
    )
    summary_table_full$variable = stringr::str_remove(summary_table_full$variable, "rvar_")

    select_rvar_beta_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_beta_original), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_sigma2 = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_intercept_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) as.numeric(chain$post_alpha_original)), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_select = rbind(
      posterior::summarize_draws(select_rvar_intercept_pool),
      posterior::summarize_draws(select_rvar_beta_pool),
      posterior::summarize_draws(select_rvar_sigma2)
    )
    summary_table_select$variable = stringr::str_remove(summary_table_select$variable, "select_rvar_")
  }else{
    rvar_beta_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_pool_beta), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_sigma2 = posterior::rvar(abind::abind(lapply(model_chains, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    rvar_intercept_pool = posterior::rvar(abind::abind(lapply(model_chains, function(chain) as.numeric(chain$post_alpha)), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_full = rbind(
      posterior::summarize_draws(rvar_intercept_pool),
      posterior::summarize_draws(rvar_beta_pool),
      posterior::summarize_draws(rvar_sigma2)
    )
    summary_table_full$variable = stringr::str_remove(summary_table_full$variable, "rvar_")

    select_rvar_beta_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_beta), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_sigma2 = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_sigma2), along = 1.5), with_chains = TRUE, nchains = nchains)
    select_rvar_intercept_pool = posterior::rvar(abind::abind(lapply(posterior_best_models, function(chain) chain$post_pool_alpha), along = 1.5), with_chains = TRUE, nchains = nchains)
    summary_table_select = rbind(
      posterior::summarize_draws(select_rvar_intercept_pool),
      posterior::summarize_draws(select_rvar_beta_pool),
      posterior::summarize_draws(select_rvar_sigma2)
    )
    summary_table_select$variable = stringr::str_remove(summary_table_select$variable, "select_rvar_")
  }

  # -------------------------------
  # 16. Give warning if unconverged
  # -------------------------------
  if (max(posterior::rhat(rvar_beta_pool), na.rm = TRUE) > 1.1) {
    warn_msg <- "pooled beta"
    warning(sprintf("Full model doesn't converge. Please increase burn-in or posterior samples. The maximum of rank normalized split-Rhat of %s is %.2f", warn_msg, max(posterior::rhat(rvar_beta_pool), na.rm = TRUE)))
  } else {
    #if (output_verbose) cat(sprintf("The maximum of rank normalized split-Rhat of %s in the full model is %.2f\n", "pooled beta", max(posterior::rhat(rvar_beta_pool), na.rm = TRUE)))
  }

  if (max(posterior::rhat(select_rvar_beta_pool), na.rm = TRUE) > 1.1) {
    warn_msg <- "pooled beta"
    warning(sprintf("Selected model doesn't converge. Please increase burn-in or posterior samples. The maximum of rank normalized split-Rhat of %s is %.2f", warn_msg, max(posterior::rhat(select_rvar_beta_pool), na.rm = TRUE)))
  } else {
    if (output_verbose) cat(sprintf("The maximum of rank normalized split-Rhat of %s in the selected model is %.2f\n", "pooled beta", max(posterior::rhat(select_rvar_beta_pool), na.rm = TRUE)))
  }


  # -------------------------------
  # 17. Handle single-chain case
  # -------------------------------
  if (length(model_chains) == 1) {
    select <- select[[1]]
    model_chains <- model_chains[[1]]
    bic_models <- bic_models[[1]]
    posterior_best_models = posterior_best_models[[1]]
    best_select <- best_select[[1]]
  }

  # -------------------------------
  # 18. Report timing
  # -------------------------------
  end <- Sys.time()
  if (output_verbose) {
    cat(sprintf("Running time for %d %s: %.2f minutes\n",
                nchains, ifelse(nchains > 1, "chains", "chain"),
                as.numeric(difftime(end, start, units = "mins"))))
  }

  return(list(posterior = model_chains, select = select, best_select = best_select, posterior_best_models = posterior_best_models, bic_models = bic_models, summary_table_full = summary_table_full, summary_table_selected = summary_table_select))
}








