// Regularized Horseshoe Bayesian MI-LASSO  (Piironen & Vehtari 2017).
// sigma2-scaled parameterization to match the package's other four models.
//
// Grouped multiply-imputed: D imputations share one set of shrinkage scalars.
//   beta_dj ~ N(0, sigma2 * tau2 * lamtilde_j^2),
//   lamtilde_j^2 = c2 * lambda_j^2 / (c2 + tau2 * lambda_j^2)
//   lambda_j ~ C+(0,1);  tau ~ C+(0, tau0), tau0 = p0/((p-p0)*sqrt(n));
//   c2 ~ Inv-Gamma(nu/2, nu*s^2/2);  y^d = alpha^d + X^d beta^d + eps, eps ~ N(0,sigma2).
//
// Sampler: slice-within-Gibbs (Neal 2003 stepping-out + shrinkage) for the
// shrinkage scalars (c2, tau2, lambda2 on the log scale); conjugate draws for
// alpha, sigma2 and the per-imputation beta blocks.
// Loop order: c2 -> tau2 -> lambda2 -> (alpha) -> sigma2 -> beta.

#include "bmi_utils.h"
#include <functional>
using namespace Rcpp;

// lamtilde_j^2 = c2 * lam2 / (c2 + tau2 * lam2), guarded away from 0.
static inline double lt_fun(double lam2, double tau2, double c2) {
  double denom = c2 + tau2 * lam2;
  double L = c2 * lam2 / denom;
  if (!(L >= 1e-300)) L = 1e-300;   // also catches NaN
  return L;
}

// Scalar slice sampler on R (Neal 2003): stepping-out (cap 50 each side) +
// shrinkage (cap 200 iters), fallback to x0. w = step width.
static inline double slice_sample(double x0,
                                  const std::function<double(double)>& logf,
                                  double w) {
  double logy = logf(x0) + std::log(R::unif_rand());
  double L = x0 - w * R::unif_rand();
  double Rt = L + w;
  int steps = 0;
  while (steps < 50 && logf(L) > logy) { L -= w; ++steps; }
  steps = 0;
  while (steps < 50 && logf(Rt) > logy) { Rt += w; ++steps; }
  for (int it = 0; it < 200; ++it) {
    double x1 = L + R::unif_rand() * (Rt - L);
    if (logf(x1) > logy) return x1;
    if (x1 < x0) L = x1; else Rt = x1;
  }
  return x0;   // fallback
}

// [[Rcpp::export]]
List cpp_reg_horseshoe(NumericVector X, NumericMatrix Yr, bool intercept,
                       double p0, double nu, double s,
                       int nburn, int npost, SEXP seed,
                       bool verbose, int printevery, int chain_index) {
  maybe_set_seed(seed);

  IntegerVector dimX = X.attr("dim");
  int D = dimX[0], n = dimX[1], p = dimX[2];

  // Clamp p0 to [1, p-1].
  if (p0 < 1.0)          p0 = 1.0;
  if (p0 > (double)p-1.0) p0 = (double)p - 1.0;
  double tau0 = p0 / (((double)p - p0) * std::sqrt((double)n));

  std::vector<arma::mat> Xd = split_X(X, D, n, p);
  arma::mat Yd(D, n);
  for (int d = 0; d < D; ++d) for (int i = 0; i < n; ++i) Yd(d, i) = Yr(d, i);

  std::vector<arma::mat> XtX(D);
  for (int d = 0; d < D; ++d) XtX[d] = Xd[d].t() * Xd[d];

  bool n_gt_p = (n > p);
  arma::mat beta;  arma::vec alpha;  double sigma2;
  pooled_init(Xd, Yd, intercept, D, n, p, n_gt_p, beta, alpha, sigma2);

  // Init shared shrinkage scalars.
  double tau2 = tau0 * tau0;
  double c2   = s * s;
  arma::vec lambda2(p, arma::fill::ones);

  arma::mat Xbeta(D, n);
  for (int d = 0; d < D; ++d) Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();
  arma::vec bm = beta_mul_cpp(beta, D, p);   // sum_d beta(d,j)^2

  std::vector<double> dbeta((size_t)npost * D * p, 0.0);
  std::vector<double> dfit((size_t)npost * D * n, 0.0);
  std::vector<double> hmp((size_t)D * n * n, 0.0);
  NumericMatrix post_lambda2(npost, p), post_alpha(npost, D);
  NumericVector post_tau2(npost), post_c2(npost), post_sigma2(npost);

  int total = nburn + npost;
  for (int it = 1; it <= total; ++it) {
    progress(verbose, chain_index, it, total, nburn, printevery);
    if (it % 200 == 0) Rcpp::checkUserInterrupt();

    // ---- 1. c2 (slice on u = log c2) --------------------------------------
    {
      auto logf = [&](double u) {
        double cc = std::exp(u);
        double lp = 0.0;
        for (int j = 0; j < p; ++j) {
          double L = lt_fun(lambda2[j], tau2, cc);
          lp += -0.5 * D * std::log(L) - 0.5 * bm[j] / (sigma2 * tau2 * L);
        }
        lp += -(nu / 2.0) * u - (nu * s * s / 2.0) / cc;   // IG(nu/2, nu s^2/2) + Jacobian
        return lp;
      };
      c2 = std::exp(slice_sample(std::log(c2), logf, 1.0));
    }

    // ---- 2. tau2 (slice on u = log tau2) ----------------------------------
    {
      auto logf = [&](double u) {
        double tt = std::exp(u);
        double lp = 0.0;
        for (int j = 0; j < p; ++j) {
          double L = lt_fun(lambda2[j], tt, c2);
          lp += -0.5 * D * std::log(tt * L) - 0.5 * bm[j] / (sigma2 * tt * L);
        }
        lp += 0.5 * u - std::log(1.0 + tt / (tau0 * tau0));  // C+(0,tau0) on tau + Jacobian
        return lp;
      };
      tau2 = std::exp(slice_sample(std::log(tau2), logf, 1.0));
    }

    // ---- 3. lambda2[j] (slice on u = log lambda2[j]) ----------------------
    for (int j = 0; j < p; ++j) {
      double bmj = bm[j];
      auto logf = [&](double u) {
        double ll = std::exp(u);
        double L = lt_fun(ll, tau2, c2);
        double lp = -0.5 * D * std::log(L) - 0.5 * bmj / (sigma2 * tau2 * L);
        lp += 0.5 * u - std::log(1.0 + ll);                  // C+(0,1) on lambda + Jacobian
        return lp;
      };
      lambda2[j] = std::exp(slice_sample(std::log(lambda2[j]), logf, 1.0));
    }

    // ---- 4. alpha (conjugate) ---------------------------------------------
    if (intercept)
      for (int d = 0; d < D; ++d) {
        double mu = arma::mean(Yd.row(d).t() - Xbeta.row(d).t());
        alpha[d] = R::rnorm(mu, std::sqrt(sigma2 / n));
      }

    // ---- 5. sigma2 (conjugate Inv-Gamma) ----------------------------------
    double SSE = 0.0;
    for (int d = 0; d < D; ++d) {
      arma::vec res = Yd.row(d).t() - Xbeta.row(d).t() - alpha[d];
      SSE += arma::dot(res, res);
    }
    double SSE_beta = 0.0;
    for (int j = 0; j < p; ++j)
      SSE_beta += bm[j] / (tau2 * lt_fun(lambda2[j], tau2, c2));
    sigma2 = rinvgamma_cpp(D * (n + p) / 2.0, (SSE + SSE_beta) / 2.0);

    // ---- 6. beta (per-imputation MVN from precision) ----------------------
    arma::vec prec_diag(p);
    for (int j = 0; j < p; ++j)
      prec_diag[j] = 1.0 / (tau2 * lt_fun(lambda2[j], tau2, c2));
    arma::mat Prec = arma::diagmat(prec_diag);
    for (int d = 0; d < D; ++d) {
      arma::mat hp;
      arma::vec r = Yd.row(d).t() - alpha[d];
      beta.row(d) = mvn_from_precision(Xd[d], XtX[d], Prec, r, sigma2, hp).t();
      Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();
      if (it > nburn)
        for (int a_ = 0; a_ < n; ++a_)
          for (int b_ = 0; b_ < n; ++b_)
            hmp[d + D * a_ + D * n * b_] += hp(a_, b_);
    }
    bm = beta_mul_cpp(beta, D, p);

    // ---- store ------------------------------------------------------------
    if (it > nburn) {
      int idx = it - nburn - 1;
      for (int j = 0; j < p; ++j) {
        post_lambda2(idx, j) = lambda2[j];
        for (int d = 0; d < D; ++d)
          dbeta[idx + (size_t)npost * d + (size_t)npost * D * j] = beta(d, j);
      }
      for (int d = 0; d < D; ++d) {
        post_alpha(idx, d) = alpha[d];
        for (int i = 0; i < n; ++i)
          dfit[idx + (size_t)npost * d + (size_t)npost * D * i] =
            Xbeta(d, i) + alpha[d] + R::rnorm(0.0, std::sqrt(sigma2));
      }
      post_tau2[idx] = tau2; post_c2[idx] = c2; post_sigma2[idx] = sigma2;
    }
  }
  for (size_t k = 0; k < hmp.size(); ++k) hmp[k] /= npost;

  List out = finalise(npost, D, p, n, dbeta, dfit, hmp);
  out["post_lambda2"] = post_lambda2;
  out["post_tau2"]    = post_tau2;
  out["post_c2"]      = post_c2;
  out["post_alpha"]   = post_alpha;
  out["post_sigma2"]  = post_sigma2;
  return out;
}
