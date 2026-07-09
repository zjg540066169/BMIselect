// =====================================================================
//  Shared helpers for the four Bayesian MI-LASSO Gibbs samplers
//  (Multi-Laplace, Horseshoe, ARD, Spike-and-Laplace).
//
//  All functions are header-only (static inline) so each translation
//  unit (per-model .cpp file) compiles a private copy with no linker
//  surprises.  Per-model files only have to include this header.
// =====================================================================
#ifndef BMI_UTILS_H
#define BMI_UTILS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>

// ----------------------------------------------------------------------
// RNG helpers (R's RNG stream so set.seed() is honoured)
// ----------------------------------------------------------------------

// Inverse-Gamma(shape a, "scale" b) matching MCMCpack::rinvgamma(n,a,b):
//   X = 1 / rgamma(shape=a, rate=b)  = 1 / R::rgamma(a, 1/b)
static inline double rinvgamma_cpp(double a, double b) {
  return 1.0 / R::rgamma(a, 1.0 / b);
}

// Inverse-Gaussian(mu, lambda) via Michael, Schucany & Haas (1976).
static inline double rinvgauss_cpp(double mu, double lambda) {
  double nu = R::norm_rand();
  double y  = nu * nu;
  double x  = mu + (mu * mu * y) / (2.0 * lambda)
              - (mu / (2.0 * lambda)) *
                std::sqrt(4.0 * mu * lambda * y + mu * mu * y * y);
  double z = R::unif_rand();
  if (z <= mu / (mu + x)) return x;
  return mu * mu / x;
}

// GIG(lambda = 1/2, chi, psi) sampler.  Every GIG draw in the four
// models uses lambda = 1/2, for which X ~ GIG(1/2, chi, psi)
// equivalently means 1/X ~ IG(mu, lam) with lam = psi, mu = sqrt(psi/chi).
// Matches GIGrvg::rgig(1, 0.5, chi, psi) in distribution.
static inline double rgig_half_cpp(double chi, double psi) {
  if (chi < 1e-12) {
    // limiting case GIG(0.5, 0, psi) = Gamma(shape=0.5, rate=psi/2)
    return R::rgamma(0.5, 2.0 / psi);
  }
  if (psi < 1e-12) psi = 1e-12;
  double mu  = std::sqrt(psi / chi);
  double lam = psi;
  double w   = rinvgauss_cpp(mu, lam);
  if (w < 1e-300) w = 1e-300;
  return 1.0 / w;
}

// ----------------------------------------------------------------------
// Linear-algebra helpers
// ----------------------------------------------------------------------

// Symmetric positive-definite inverse with graceful fallback.
static inline arma::mat spdinv_cpp(const arma::mat& A) {
  arma::mat Ainv;
  bool ok = arma::inv_sympd(Ainv, A);
  if (!ok) {
    if (!arma::inv(Ainv, A)) Ainv = arma::pinv(A);
  }
  return Ainv;
}

// Draw beta_d ~ MVN( va * Xd^T r , sigma2 * va ),  va = (XtX + Prec)^{-1}
// and return hatproj_d = Xd * va * Xd^T (n x n).  z drawn via R RNG.
static inline arma::vec mvn_from_precision(const arma::mat& Xd,
                                           const arma::mat& XtX,
                                           const arma::mat& Prec,
                                           const arma::vec& r,
                                           double sigma2,
                                           arma::mat& hatproj_out) {
  arma::mat A  = XtX + Prec;
  arma::mat va = spdinv_cpp(A);
  va = 0.5 * (va + va.t());                 // enforce symmetry
  arma::vec mu = va * (Xd.t() * r);
  hatproj_out  = Xd * va * Xd.t();

  arma::mat Sig = sigma2 * va;
  Sig = 0.5 * (Sig + Sig.t());
  arma::mat L;
  bool ok = arma::chol(L, Sig, "lower");
  if (!ok) {
    double jit = 1e-8 * (arma::trace(Sig) / Sig.n_rows + 1e-12);
    arma::mat I = arma::eye(Sig.n_rows, Sig.n_rows);
    while (!arma::chol(L, Sig + jit * I, "lower") && jit < 1e3) jit *= 10.0;
    if (L.n_rows == 0) L = arma::diagmat(arma::sqrt(arma::abs(Sig.diag())));
  }
  arma::vec z(mu.n_elem);
  for (arma::uword k = 0; k < z.n_elem; ++k) z[k] = R::norm_rand();
  return mu + L * z;
}

// Build per-imputation design matrices from an R array X with dim (D,n,p).
static inline std::vector<arma::mat> split_X(const Rcpp::NumericVector& X,
                                             int D, int n, int p) {
  std::vector<arma::mat> Xd(D, arma::mat(n, p));
  for (int d = 0; d < D; ++d)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < p; ++j)
        Xd[d](i, j) = X[d + D * i + D * n * j];
  return Xd;
}

// OLS-based initialisation mirroring pooledResidualVariance() in the R code.
static inline void pooled_init(const std::vector<arma::mat>& Xd,
                               const arma::mat& Yd,        // D x n
                               bool intercept, int D, int n, int p,
                               bool n_gt_p,
                               arma::mat& beta_out,        // D x p
                               arma::vec& alpha_out,       // D
                               double& sigma2_out) {
  beta_out.zeros(D, p);
  alpha_out.zeros(D);
  arma::vec resvar(D);
  for (int d = 0; d < D; ++d) {
    arma::vec y = Yd.row(d).t();
    if (!n_gt_p) {
      double m = arma::mean(y);
      arma::vec res = y - (intercept ? m : 0.0);
      int kdf = intercept ? 1 : 0;
      resvar[d] = arma::dot(res, res) / std::max(1, n - kdf);
      continue;
    }
    arma::mat Xdesign;
    if (intercept) {
      Xdesign = arma::join_rows(arma::ones(n), Xd[d]);
    } else {
      Xdesign = Xd[d];
    }
    arma::vec coef = arma::solve(Xdesign, y);
    arma::vec res  = y - Xdesign * coef;
    int k = Xdesign.n_cols;
    resvar[d] = arma::dot(res, res) / std::max(1, n - k);
    if (intercept) {
      alpha_out[d] = coef[0];
      for (int j = 0; j < p; ++j) beta_out(d, j) = coef[j + 1];
    } else {
      for (int j = 0; j < p; ++j) beta_out(d, j) = coef[j];
    }
  }
  sigma2_out = arma::mean(resvar);
}

// sum_d beta(d, j)^2  for j = 1..p
static inline arma::vec beta_mul_cpp(const arma::mat& beta, int D, int p) {
  arma::vec bm(p);
  for (int j = 0; j < p; ++j) {
    double s = 0.0;
    for (int d = 0; d < D; ++d) s += beta(d, j) * beta(d, j);
    bm[j] = s;
  }
  return bm;
}

// Common output assembler.  draws_beta is flat column-major length npost*D*p
// with index it + npost*d + npost*D*j  ->  post_beta dim (npost,D,p).
static inline Rcpp::List finalise(int npost, int D, int p, int n,
                                  std::vector<double>& draws_beta,
                                  std::vector<double>& draws_fit,
                                  std::vector<double>& hatproj) {
  Rcpp::NumericVector post_beta(draws_beta.begin(), draws_beta.end());
  post_beta.attr("dim") = Rcpp::IntegerVector::create(npost, D, p);
  Rcpp::NumericVector post_pool_beta(draws_beta.begin(), draws_beta.end());
  post_pool_beta.attr("dim") = Rcpp::IntegerVector::create(npost * D, p);

  Rcpp::NumericVector post_fit(draws_fit.begin(), draws_fit.end());
  post_fit.attr("dim") = Rcpp::IntegerVector::create(npost, D, n);
  Rcpp::NumericVector post_pool_fit(draws_fit.begin(), draws_fit.end());
  post_pool_fit.attr("dim") = Rcpp::IntegerVector::create(npost * D, n);

  Rcpp::NumericVector hmp(hatproj.begin(), hatproj.end());
  hmp.attr("dim") = Rcpp::IntegerVector::create(D, n, n);

  return Rcpp::List::create(
    Rcpp::_["post_beta"]            = post_beta,
    Rcpp::_["post_fitted_Y"]        = post_fit,
    Rcpp::_["post_pool_beta"]       = post_pool_beta,
    Rcpp::_["post_pool_fitted_Y"]   = post_pool_fit,
    Rcpp::_["hat_matrix_proj"]      = hmp);
}

static inline void maybe_set_seed(SEXP seed) {
  if (seed != R_NilValue) {
    Rcpp::Environment base = Rcpp::Environment::base_env();
    Rcpp::Function set_seed = base["set.seed"];
    set_seed(seed);
  }
}

static inline void progress(bool verbose, int chain, int it, int total,
                            int nburn, int printevery) {
  if (verbose && printevery > 0 && it % printevery == 0)
    Rcpp::Rcout << "Chain " << chain << ": " << it << "/" << total << ", "
                << (it <= nburn ? "burn-in" : "sampling") << "\n";
}

#endif  // BMI_UTILS_H
