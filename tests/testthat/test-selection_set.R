# Tests for the `selection_set` argument of BMI_LASSO(): supplying a fixed
# logical selection set must bypass the four-step search (candidate generation
# + BIC) and project the fitted posterior directly onto that set -- identical to
# a manual projection_posterior() call on the same draws. This underpins
# post-selection inference under a fixed (e.g. the true) model.

test_that("selection_set input is validated before fitting", {
  d <- sim_A(n = 50, p = 20, type = "MAR", SNP = 1.5,
             low_missing = TRUE, n_imp = 3, seed = 2)
  X <- d$data_MI$X; Y <- d$data_MI$Y

  # non-logical input -> error (raised before any MCMC runs)
  expect_error(
    BMI_LASSO(X, Y, model = "Multi_Laplace", selection_set = c(1, 5, 11),
              nburn = 20, npost = 20, seed = 1, output_verbose = FALSE),
    "logical vector of length"
  )
  # logical vector of the wrong length -> error
  expect_error(
    BMI_LASSO(X, Y, model = "Multi_Laplace", selection_set = rep(TRUE, 5),
              nburn = 20, npost = 20, seed = 1, output_verbose = FALSE),
    "logical vector of length"
  )
})

test_that("selection_set bypasses BIC and matches manual projection", {
  skip_on_cran()  # runs MCMC

  d <- sim_A(n = 100, p = 20, type = "MAR", SNP = 1.5,
             low_missing = TRUE, n_imp = 5, seed = 1)
  X <- d$data_MI$X; Y <- d$data_MI$Y
  S <- d$important                       # logical length-p true active set

  # standardize = FALSE so the internal design equals X (lets us replicate
  # the internal projection externally, without the standardisation transform).
  f <- BMI_LASSO(X, Y, model = "Multi_Laplace", selection_set = S,
                 standardize = FALSE, nburn = 200, npost = 200,
                 seed = 7, output_verbose = FALSE)

  # best_select equals the supplied set; the search/BIC step is skipped.
  expect_setequal(which(as.logical(f$best_select)), which(S))
  expect_null(f$bic_models)

  # the projected posterior equals a manual projection of the SAME posterior
  # draws onto S -- i.e. selection_set truly routes through projection_posterior.
  manual <- projection_posterior(X, f$posterior$post_beta, f$posterior$post_sigma2,
                                 matrix(S, nrow = 1),
                                 alpha1_arr = f$posterior$post_alpha)
  expect_equal(f$posterior_best_models$post_beta,   manual$beta2_arr)
  expect_equal(f$posterior_best_models$post_sigma2, manual$sigma2_opt)

  # projected coefficients vanish outside the supplied set.
  pm <- colMeans(f$posterior_best_models$post_pool_beta)
  expect_true(all(which(abs(pm) > 1e-8) %in% which(S)))

  # running_time is returned as a non-negative numeric (minutes).
  expect_true(is.numeric(f$running_time) && f$running_time >= 0)
})
