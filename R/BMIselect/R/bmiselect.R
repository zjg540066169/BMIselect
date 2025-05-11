#source("R/ARD.R")
#source("R/Horseshoe.R")
#source("R/Multi_Laplace.R")
#source("R/Spike_Normal.R")
#source("R/Spike_Laplace.R")
#source("R/utils.R")




#' Bayesian MI Variable Selection via MCMC
#'
#' \code{bmiselect} implements Bayesian variable selection across multiply-imputed datasets
#' using either shrinkage-based priors or spike-and-slab priors. You can choose among
#' five models and obtain posterior draws, and (optionally) BIC-based best model selection.
#'
#' @param X A \code{D x n x p} array or \code{n x p} matrix of predictors.
#'   If \code{X} is a matrix, it is recycled into an array with \code{D = 1}.
#' @param Y A \code{D x n} matrix or vector of outcomes. If \code{Y} is a vector/matrix
#'   of length \code{n}, it is recycled with \code{D = 1}.
#' @param model Character. Which model to run; one of
#'   \code{"Multi_Laplace"}, \code{"Horseshoe"}, \code{"ARD"},
#'   \code{"Spike_Normal"}, or \code{"Spike_Laplace"}.
#' @param standardize Logical; if \code{TRUE}, center & scale each column of \code{X}
#'   within each dataset before MCMC.
#' @param shrinkage_sym_CI Numeric vector of credible-interval levels (e.g.
#'   \code{seq(0.05,0.95,by=0.05)}) for shrinkage-based models. Only used when \code{model}
#'   is one of \code{"Multi_Laplace"}, \code{"Horseshoe"}, \code{"ARD"}.
#' @param spike_slab_threshold Numeric vector of posterior-median thresholds for
#'   spike-and-slab models. Only used when \code{model}
#'   is either \code{"Spike_Normal"} or \code{"Spike_Laplace"}.
#' @param bic_select Logical; if \code{TRUE}, perform BIC-based selection for
#'   the MCMC draws.
#' @param nburn Integer number of burn-in iterations.
#' @param npost Integer number of posterior draws to retain after burn-in.
#' @param seed Optional integer seed or \code{NULL} for reproducible chains.
#' @param nchain Integer number of independent MCMC chains.
#' @param ncores Integer number of parallel cores; if \code{1}, runs sequentially.
#' @param verbose Logical; whether to print progress messages.
#' @param printevery Integer; print progress every \code{printevery} MCMC iterations.
#' @param ... Additional model-specific arguments:
#'   \describe{
#'     \item{h}{default \code{2}; shrinkage parameter for \code{Multi_Laplace}}
#'     \item{s}{default \code{(D + 1)/D}; scale hyperparameter for \code{Multi_Laplace}}
#'     \item{v02}{default \code{2}; slab variance for \code{Spike_Normal}}
#'     \item{p0}{default \code{0.5}; prior inclusion probability for \code{Spike_Normal}}
#'     \item{a}{default \code{2}; shape parameter for \code{Spike_Laplace}}
#'     \item{b}{default \code{2 * (D + 1)/D}; scale parameter for \code{Spike_Laplace}}
#'     \item{prop_sd}{default \code{1}; SD of proposal lognormal distribution in M-H step for \code{Spike_Laplace}}
#'   }
#'
#' @return A list with components:
#'   \describe{
#'     \item{posterior}{
#'       If \code{nchain = 1}, a list of posterior draws of parameters (vectors or matrices);
#'       if \code{nchain > 1}, a list of such lists, one per chain.
#'     }
#'     \item{select}{
#'       Logical inclusion indicators: if \code{nchain = 1}, a matrix of TRUE/FALSE values for each criterion and variable;
#'       if \code{nchain > 1}, a list of such matrices, one per chain.
#'     }
#'     \item{best_select}{
#'       When \code{bic_select = TRUE}, the variable selection at the best BIC threshold:
#'       if \code{nchain = 1}, a logical vector;
#'       if \code{nchain > 1}, a list of logical vectors, one per chain.
#'     }
#'   }
#'
#' @examples
#' # simulate 5 imputed datasets, n = 100, p = 10
#' X <- array(rnorm(5 * 100 * 10), dim = c(5, 100, 10))
#' Y <- matrix(rnorm(5 * 100), nrow = 5, ncol = 100)
#'
#' # run two chains in parallel
#' out <- bmiselect(
#'   X, Y,
#'   model = "Horseshoe",
#'   standardize = TRUE,
#'   shrinkage_sym_CI = seq(0.05, 0.95, by = 0.05),
#'   bic_select = TRUE,
#'   nburn = 1000,
#'   npost = 1000,
#'   nchain = 2,
#'   ncores = 1,
#'   seed = 1,
#'   printevery = 200
#' )
#' str(out)
#'
#' @export
#'
#' @importFrom stats as.formula coef dgamma dlnorm gaussian glm lm median plogis predict quantile rbeta rbinom rcauchy residuals rgamma rlnorm rnorm runif sd var vcov


bmiselect = function(X, Y, model, standardize = FALSE, shrinkage_sym_CI = seq(0.05, 0.95, by = 0.05), spike_slab_threshold = seq(0.05, 0.95, by = 0.05), bic_select = TRUE, nburn = 4000, npost = 4000, seed = NULL, nchain = 1, ncores = 1, verbose = T, printevery = 1000, ...){
  # -------------------------------
  # 1. Validate input model
  # -------------------------------
  if (!model %in% c("Multi_Laplace", "Horseshoe", "ARD", "Spike_Normal", "Spike_Laplace")) {
    stop("Invalid model_name. Available options: Multi_Laplace, Horseshoe, ARD, Spike_Normal, Spike_Laplace.")
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
    X_norm = list()
    for (d in 1:D) {
      x = X[d,,]
      X_mean[[d]] = apply(x, 2, mean)
      X_norm[[d]] = apply(x, 2, sd)
      X[d,,] = scale(x)
    }
  }

  # -------------------------------
  # 6. Handle model-specific parameters from ...
  # -------------------------------
  extra_parameters <- list(...)
  switch(model,
         "Multi_Laplace" = {
           if (!("h" %in% names(extra_parameters))) {
             if(verbose) cat("Missing 'h' for Multi_Laplace. Use default value h = 2", "\n")
             extra_parameters$h = 2
           }
           if (!("s" %in% names(extra_parameters))) {
             if(verbose) cat(paste0("Missing 's' for Multi_Laplace. Use default value s = ", (D+1)/D), "\n")
             extra_parameters$s = NULL
           }
         },
         "Spike_Normal" = {
           if (!("v02" %in% names(extra_parameters))) {
             if(verbose) cat("Missing 'v02' for Spike_Normal. Use default value v02 = 2", "\n")
             extra_parameters$v02 = 2
           }
           if (!("p0" %in% names(extra_parameters))) {
             if(verbose) cat("Missing 'p0' for Spike_Normal. Use default value p0 = 0.5", "\n")
             extra_parameters$p0 = 0.5
           }
         },
         "Spike_Laplace" = {
           if (!("a" %in% names(extra_parameters))) {
             if(verbose) cat("Missing 'a' for Spike_Laplace. Use default value a = 2", "\n")
             extra_parameters$a = 2
           }
           if (!("b" %in% names(extra_parameters))) {
             if(verbose) cat(paste0("Missing 'b' for Spike_Laplace. Use default value b = ", 2 * (D+1)/D), "\n")
             extra_parameters$b = NULL
           }
           if (!("prop_sd" %in% names(extra_parameters))) {
             if(verbose) cat("Missing 'prop_sd' for Spike_Laplace. Use default value prop_sd = 1\n")
             extra_parameters$prop_sd = 1
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
    chain = 1:nchain, .combine = list, .multicombine = TRUE,
    .maxcombine = ifelse(nchain >= 2, nchain, 2)
  ) %dopar% {
    seed_chain = if (!is.null(seed)) seed + chain else NULL
    return(switch(model,
                  "Multi_Laplace" = {
                    multi_laplace_mcmc(X, Y, h = extra_parameters$h, s = extra_parameters$s,
                                       nburn = nburn, npost = npost, seed = seed_chain,
                                       verbose = verbose, printevery = printevery, chain_index = chain)
                  },
                  "Horseshoe" = {
                    horseshoe_mcmc(X, Y, standardize = standardize,
                                   nburn = nburn, npost = npost, seed = seed_chain,
                                   verbose = verbose, printevery = printevery, chain_index = chain)
                  },
                  "ARD" = {
                    ARD_mcmc(X, Y, standardize = standardize,
                             nburn = nburn, npost = npost, seed = seed_chain,
                             verbose = verbose, printevery = printevery, chain_index = chain)
                  },
                  "Spike_Normal" = {
                    spike_normal_mcmc(X, Y, standardize = standardize,
                                      v02 = extra_parameters$v02, p0 = extra_parameters$p0,
                                      nburn = nburn, npost = npost, seed = seed_chain,
                                      verbose = verbose, printevery = printevery, chain_index = chain)
                  },
                  "Spike_Laplace" = {
                    spike_laplace_mcmc(X, Y, standardize = standardize,
                                       a = extra_parameters$a, b = extra_parameters$b, prop_sd = extra_parameters$prop_sd,
                                       nburn = nburn, npost = npost, seed = seed_chain,
                                       verbose = verbose, printevery = printevery, chain_index = chain)
                  }
    ))
  })

  # -------------------------------
  # 9. Reset parallel backend
  # -------------------------------
  if (ncores > 1) {
    doParallel::stopImplicitCluster()
    foreach::registerDoSEQ()
  }

  # -------------------------------
  # 10. Handle degenerate list case
  # -------------------------------
  if ("post_sigma2" %in% names(model_chains)) {
    model_chains = list(model_chains)
  }


  # -------------------------------
  # 11. Perform variable selection
  # - For shrinkage models: use symmetric credible intervals
  # - For spike-and-slab: use posterior median threshold
  # -------------------------------
  if (model %in% c("Multi_Laplace", "Horseshoe", "ARD")) {
    select = lapply(model_chains, function(c) {
      t(sapply(as.character(shrinkage_sym_CI), function(ci) {
        ci = as.numeric(ci)
        apply(c$post_pool_beta, 2, function(x_j)
          quantile(x_j, (1 - ci)/2) > 0 | quantile(x_j, 1 - (1 - ci)/2) < 0)
      }, simplify = TRUE, USE.NAMES = TRUE))
    })
  } else if (model == "Spike_Normal") {
    select = lapply(model_chains, function(c) {
      t(sapply(as.character(spike_slab_threshold), function(ci) {
        ci = as.numeric(ci)
        apply(c$post_gamma, 2, function(x_j) median(x_j) >= ci)
      }, simplify = TRUE, USE.NAMES = TRUE))
    })
  } else {
    select = lapply(model_chains, function(c) {
      t(sapply(as.character(spike_slab_threshold), function(ci) {
        ci = as.numeric(ci)
        apply(c$post_theta, 2, function(x_j) median(x_j) >= ci)
      }, simplify = TRUE, USE.NAMES = TRUE))
    })
  }


  # -------------------------------
  # 12. Undo standardization (if requested)
  # Convert posterior draws back to original scale
  # -------------------------------
  if (standardize == TRUE) {
    for (chain in 1:nchain) {
      model_chains[[chain]][["post_beta_original"]] = array(NA, dim = c(npost, D, p))
      model_chains[[chain]][["post_pool_beta_original"]] = matrix(NA, nrow = npost * D, ncol = p)
      for (j in 1:p) {
        model_chains[[chain]][["post_beta_original"]][,,j] =
          sapply(1:D, function(d) model_chains[[chain]][["post_beta"]][,d,j] / X_norm[[d]][j])
        model_chains[[chain]][["post_pool_beta_original"]][,j] =
          model_chains[[chain]][["post_beta_original"]][,,j]
      }
    }
  }

  # -------------------------------
  # 13. Handle single-chain case
  # -------------------------------
  if (length(model_chains) == 1) {
    select <- select[[1]]
    model_chains <- model_chains[[1]]

    if (model == "Spike_Laplace") {
      cat("The average acceptance ratio for lambda2 is", mean(model_chains$post_accept_ratio), "\n")
    }

  } else {
    # -------------------------------
    # 14. Multiple chains: check convergence with Gelman-Rubin
    # -------------------------------
    varname <- if (model == "Spike_Normal") "post_logit_inclusion_prob" else "post_pool_beta"

    # Extract posterior draws and wrap into coda::mcmc.list
    chain_draws <- lapply(model_chains, function(chain) chain[[varname]])
    combined_chains <- coda::mcmc.list(lapply(chain_draws, coda::mcmc))

    # Compute multivariate R-hat, fallback to univariate if needed
    rhat <- tryCatch({
      coda::gelman.diag(combined_chains, multivariate = TRUE)$mpsrf
    }, error = function(e) {
      if (model != "Spike_Normal")
        warning("Multivariate PSRF failed, switching to univariate PSRF.")
      max(coda::gelman.diag(combined_chains, multivariate = FALSE)$psrf[, "Point est."], na.rm = TRUE)
    })

    # Report convergence warning or info
    if (rhat > 1.1) {
      warn_msg <- if (model == "Spike_Normal") "logit inclusion probabilities" else "pooled beta"
      warning(sprintf("Multiple chains don't converge. Please increase burn-in or posterior samples. The R-hat of %s is %.3f", warn_msg, rhat))
    } else {
      if (verbose) cat(sprintf("The R-hat of %s is %.3f\n", if (model == "Spike_Normal") "inclusion probability" else "pooled beta", rhat))
    }

    # -------------------------------
    # 15. Re-check convergence on original scale (if standardized)
    # -------------------------------
    if (standardize == TRUE && model != "Spike_Normal") {
      varname <- "post_pool_beta_original"
      chain_draws <- lapply(model_chains, function(chain) chain[[varname]])
      combined_chains <- coda::mcmc.list(lapply(chain_draws, coda::mcmc))

      rhat <- tryCatch({
        coda::gelman.diag(combined_chains, multivariate = TRUE)$mpsrf
      }, error = function(e) {
        warning("Multivariate PSRF failed, switching to univariate PSRF.")
        max(coda::gelman.diag(combined_chains, multivariate = FALSE)$psrf[, "Point est."], na.rm = TRUE)
      })

      if (rhat > 1.1) {
        warning(sprintf("Multiple chains don't converge. R-hat of pooled beta (original scale) is %.3f", rhat))
      } else {
        if (verbose) cat(sprintf("The R-hat of pooled beta on the original scale is %.3f\n", rhat))
      }
    }

    # -------------------------------
    # 16. Check for selection consistency across chains
    # -------------------------------
    if (model %in% c("Multi_Laplace", "Horseshoe", "ARD")) {
      inconsistent_criteria <- sapply(shrinkage_sym_CI, function(i) {
        !all(colMeans(do.call(rbind, lapply(select, function(sl) sl[as.character(i), ]))) %in% c(0, 1))
      })
      if (any(inconsistent_criteria)) {
        warning(sprintf("Multiple chains produce inconsistent selected variables for credible interval %s. Please check selected variables carefully, and increase numbers of MCMC iterations.",
                        paste(shrinkage_sym_CI[inconsistent_criteria], collapse = ", ")))
      }
    } else {
      inconsistent_criteria <- sapply(spike_slab_threshold, function(i) {
        !all(colMeans(do.call(rbind, lapply(select, function(sl) sl[as.character(i), ]))) %in% c(0, 1))
      })
      if (any(inconsistent_criteria)) {
        warning(sprintf("Multiple chains produce inconsistent selected variables for posterior median threshold %s. Please check selected variables carefully, and increase numbers of MCMC iterations.",
                        paste(spike_slab_threshold[inconsistent_criteria], collapse = ", ")))
      }
    }
  }

  # -------------------------------
  # 17. If BIC selection is requested
  # -------------------------------
  if (bic_select) {
    if (is.null(dim(select))) {
      # Multiple chains: list of selection matrices
      bic_models <- lapply(select, function(l) {
        compute_bic_glm_fixed_sigma2(l, X_O, Y_O,
                                     median(do.call(c, lapply(model_chains, function(m) m$post_sigma2))))
      })
    } else {
      # Single chain
      bic_models <- compute_bic_glm_fixed_sigma2(select, X_O, Y_O, median(model_chains$post_sigma2))
    }
  }

  # -------------------------------
  # 18. Report timing
  # -------------------------------
  end <- Sys.time()
  if (verbose) {
    cat(sprintf("Running time for %d %s: %.2f minutes\n",
                nchain, ifelse(nchain > 1, "chains", "chain"),
                as.numeric(difftime(end, start, units = "mins"))))
  }

  # -------------------------------
  # 19. Return results
  # -------------------------------
  if (bic_select) {
    if (is.null(dim(select))) {
      best_select <- sapply(seq_along(bic_models), function(i) {
        select[[i]][which.min(bic_models[[i]]), , drop = FALSE]
      }, simplify = FALSE)
      return(list(posterior = model_chains, select = select, best_select = best_select))
    } else {
      best_select <- select[which.min(bic_models), , drop = FALSE]
      rownames(best_select) <- rownames(select)[which.min(bic_models)]
      return(list(posterior = model_chains, select = select, best_select = best_select))
    }
  }

  return(list(posterior = model_chains, select = select))
}








