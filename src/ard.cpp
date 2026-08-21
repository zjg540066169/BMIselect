// ARD Bayesian MI-LASSO  (Web Algorithm 3 of supp.tex).
// Loop order: (alpha) -> sigma2 -> psi2 -> beta.
// psi2_j is capped at 1/eps (eps = 1e-6) to prevent over/underflow when
// beta_mul_j is tiny -- matches the safeguard in the reference R sampler.

#include "bmi_utils.h"
using namespace Rcpp;

// 1-D slice sampler for the MARGINAL full conditional of an ARD precision psi_j
// with its D coefficients beta_{.,j} integrated out (approach A). Sampled on
// t = log(psi) so the log target (Jacobian absorbed) is
//   logf(t) = (D/2) t - 0.5 sum_d log(c_d + e^t)
//                     + sum_d z_d^2 / (2 sigma2 (c_d + e^t)),
// where c_d = X_{d,j}'X_{d,j} and z_d = X_{d,j}'(residual with coord j added back).
// psi is truncated to (0, 1/eps], matching the reference sampler's cap. Neal (2003)
// stepping-out + shrinkage slice; leaves the exact (truncated) target invariant.
static inline double slice_psi_ard(double psi_cur, const arma::vec& cvec,
                                   const arma::vec& zvec, double sigma2,
                                   int D, double eps) {
  const double logcap = std::log(1.0 / eps);
  const double tmin   = -20.0;                         // psi >= ~2e-9
  auto logf = [&](double t) -> double {
    double psi = std::exp(t), val = 0.5 * D * t;
    for (int d = 0; d < D; ++d) {
      double cp = cvec[d] + psi;
      val += -0.5 * std::log(cp) + zvec[d] * zvec[d] / (2.0 * sigma2 * cp);
    }
    return val;
  };
  double t0 = std::log(std::min(std::max(psi_cur, std::exp(tmin)), 1.0 / eps));
  double y  = logf(t0) + std::log(R::unif_rand());     // slice level
  double w  = 1.0;
  double L  = t0 - w * R::unif_rand(), Rr = L + w;
  for (int k = 0; k < 50 && L  > tmin   && logf(L)  > y; ++k) L  -= w;
  for (int k = 0; k < 50 && Rr < logcap && logf(Rr) > y; ++k) Rr += w;
  if (L  < tmin)   L  = tmin;
  if (Rr > logcap) Rr = logcap;
  for (int k = 0; k < 100; ++k) {
    double t = L + R::unif_rand() * (Rr - L);
    if (logf(t) > y) return std::exp(t);
    if (t < t0) L = t; else Rr = t;
  }
  return psi_cur;                                      // fallback (numerical guard)
}

// [[Rcpp::export]]
List cpp_ard(NumericVector X, NumericMatrix Yr, bool intercept,
             int nburn, int npost, SEXP seed,
             bool verbose, int printevery, int chain_index) {
  maybe_set_seed(seed);

  IntegerVector dimX = X.attr("dim");
  int D = dimX[0], n = dimX[1], p = dimX[2];
  const double eps = 1e-6;

  std::vector<arma::mat> Xd = split_X(X, D, n, p);
  arma::mat Yd(D, n);
  for (int d = 0; d < D; ++d) for (int i = 0; i < n; ++i) Yd(d, i) = Yr(d, i);

  std::vector<arma::mat> XtX(D);
  for (int d = 0; d < D; ++d) XtX[d] = Xd[d].t() * Xd[d];

  bool n_gt_p = (n > p);
  arma::mat beta;  arma::vec alpha;  double sigma2;
  pooled_init(Xd, Yd, intercept, D, n, p, n_gt_p, beta, alpha, sigma2);

  // ARD's psi2_j = Gamma(., 2 sigma2 / beta_mul_j) diverges to the 1/eps cap when
  // beta_j starts at 0, which permanently pins beta_j at 0 (a self-reinforcing
  // trap unique to ARD's precision update). pooled_init returns beta = 0 when
  // p >= n, so seed ARD there with a ridge estimate instead, so every coefficient
  // starts away from 0. (No effect when n > p, where beta is the OLS fit.)
  if (!n_gt_p) {
    double ridge = (double) n;                       // moderate; X standardized -> diag(X'X) ~ n
    for (int d = 0; d < D; ++d) {
      arma::vec y = Yd.row(d).t();
      if (intercept) { alpha[d] = arma::mean(y); y -= alpha[d]; }
      arma::mat A = XtX[d]; A.diag() += ridge;
      beta.row(d) = arma::solve(A, Xd[d].t() * y).t();
    }
  }

  arma::mat Xbeta(D, n);
  for (int d = 0; d < D; ++d) Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();
  arma::vec beta_mul = beta_mul_cpp(beta, D, p);

  arma::vec psi2(p);
  for (int j = 0; j < p; ++j) {
    psi2[j] = R::rgamma(D / 2.0, 2.0 * sigma2 / beta_mul[j]);
    psi2[j] = std::min(1.0 / eps, psi2[j]);
  }

  std::vector<double> dbeta((size_t)npost * D * p, 0.0);
  std::vector<double> dfit((size_t)npost * D * n, 0.0);
  std::vector<double> hmp((size_t)D * n * n, 0.0);
  NumericMatrix post_psi2(npost, p), post_alpha(npost, D);
  NumericVector post_sigma2(npost);

  int total = nburn + npost;
  for (int it = 1; it <= total; ++it) {
    progress(verbose, chain_index, it, total, nburn, printevery);
    if (it % 200 == 0) Rcpp::checkUserInterrupt();

    if (intercept)
      for (int d = 0; d < D; ++d) {
        double mu = arma::mean(Yd.row(d).t() - Xbeta.row(d).t());
        alpha[d] = R::rnorm(mu, std::sqrt(sigma2 / n));
      }

    double SSE = 0.0;
    for (int d = 0; d < D; ++d) {
      arma::vec res = Yd.row(d).t() - Xbeta.row(d).t() - alpha[d];
      SSE += arma::dot(res, res);
    }
    double SSE_beta = 0.0;
    for (int j = 0; j < p; ++j) SSE_beta += beta_mul[j] * psi2[j];
    sigma2 = rinvgamma_cpp(D * (n + p) / 2.0, (SSE + SSE_beta) / 2.0);

    // ---- Collapsed coordinate scan (approach A): break the psi-beta funnel ----
    // For each coordinate j, draw psi2[j] from its MARGINAL full conditional
    // (its D coefficients beta_{.,j} integrated out) via a 1-D slice, then draw
    // beta_{d,j} | psi2[j] in closed form. Sampling psi2[j] free of the current
    // beta_{.,j} lets it jump across the funnel neck instead of crawling with it.
    // This is an exact, partially-collapsed Gibbs scan on the SAME target as the
    // block sampler (each (psi_j, beta_{.,j}) drawn from its joint conditional).
    std::vector<arma::vec> eres(D);
    for (int d = 0; d < D; ++d) eres[d] = Yd.row(d).t() - alpha[d] - Xbeta.row(d).t();
    for (int j = 0; j < p; ++j) {
      arma::vec cvec(D), zvec(D);
      for (int d = 0; d < D; ++d) {
        double cdj = XtX[d](j, j);
        cvec[d] = cdj;
        zvec[d] = arma::dot(Xd[d].col(j), eres[d]) + cdj * beta(d, j); // X_{d,j}'(resid incl. j)
      }
      psi2[j] = slice_psi_ard(psi2[j], cvec, zvec, sigma2, D, eps);
      for (int d = 0; d < D; ++d) {
        double prec = cvec[d] + psi2[j];
        double bnew = R::rnorm(zvec[d] / prec, std::sqrt(sigma2 / prec));
        eres[d] -= Xd[d].col(j) * (bnew - beta(d, j));
        beta(d, j) = bnew;
      }
    }
    for (int d = 0; d < D; ++d) Xbeta.row(d) = (Yd.row(d).t() - alpha[d] - eres[d]).t();
    beta_mul = beta_mul_cpp(beta, D, p);

    // hat matrix (posterior mean, for the four-step projection df) depends only on
    // (X_d, psi2). Use spdinv_cpp (fork-safe, as in mvn_from_precision), NOT
    // arma::inv_sympd: the latter calls LAPACK directly and segfaults inside a
    // mclapply fork worker on macOS Accelerate (GCD + fork), under ncores>1.
    if (it > nburn)
      for (int d = 0; d < D; ++d) {
        arma::mat va = spdinv_cpp(XtX[d] + arma::diagmat(psi2));
        va = 0.5 * (va + va.t());
        arma::mat hp = Xd[d] * va * Xd[d].t();
        for (int a_ = 0; a_ < n; ++a_)
          for (int b_ = 0; b_ < n; ++b_)
            hmp[d + D * a_ + D * n * b_] += hp(a_, b_);
      }

    if (it > nburn) {
      int idx = it - nburn - 1;
      for (int j = 0; j < p; ++j) {
        post_psi2(idx, j) = psi2[j];
        for (int d = 0; d < D; ++d)
          dbeta[idx + (size_t)npost * d + (size_t)npost * D * j] = beta(d, j);
      }
      for (int d = 0; d < D; ++d) {
        post_alpha(idx, d) = alpha[d];
        for (int i = 0; i < n; ++i)
          dfit[idx + (size_t)npost * d + (size_t)npost * D * i] =
            Xbeta(d, i) + alpha[d] + R::rnorm(0.0, std::sqrt(sigma2));
      }
      post_sigma2[idx] = sigma2;
    }
  }
  for (size_t k = 0; k < hmp.size(); ++k) hmp[k] /= npost;

  List out = finalise(npost, D, p, n, dbeta, dfit, hmp);
  out["post_psi2"]   = post_psi2;
  out["post_alpha"]  = post_alpha;
  out["post_sigma2"] = post_sigma2;
  return out;
}
