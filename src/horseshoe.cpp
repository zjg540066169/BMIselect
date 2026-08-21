// Horseshoe Bayesian MI-LASSO  (Web Algorithm 2 of supp.tex).
// Auxiliary inverse-gamma representation of half-Cauchy (Makalic & Schmidt 2016).
// Loop order: eta -> tau2 -> kappa -> lambda2 -> (alpha) -> sigma2 -> beta.

#include "bmi_utils.h"
using namespace Rcpp;

// 1-D slice sampler for the MARGINAL full conditional of a Horseshoe local scale
// lambda2_j with its D coefficients beta_{.,j} integrated out (phi_j = 1/(lambda2_j
// tau2) is the prior precision). On u = log(lambda2) the log target (Jacobian
// absorbed) is
//   logf(u) = -(D+1)/2 u - e^{-u}/kappa_j
//                        - 0.5 sum_d log(c_d + phi) + sum_d z_d^2/(2 sigma2 (c_d + phi)),
// with phi = 1/(e^u tau2), c_d = X_{d,j}'X_{d,j}, z_d = X_{d,j}'(resid incl. j). The
// prior term -(1/2)log(lambda2) - 1/(kappa_j lambda2) is the InvGamma(1/2, 1/kappa_j)
// half-Cauchy augmentation. Neal (2003) stepping-out slice; exact.
static inline double slice_lam2_hs(double lam2_cur, double kappa_j, double tau2,
                                   double sigma2, const arma::vec& cvec,
                                   const arma::vec& zvec, int D) {
  auto logf = [&](double u) -> double {
    double lam2 = std::exp(u), phi = 1.0 / (lam2 * tau2);
    double val = -0.5 * (D + 1.0) * u - std::exp(-u) / kappa_j;
    for (int d = 0; d < D; ++d) {
      double cp = cvec[d] + phi;
      val += -0.5 * std::log(cp) + zvec[d] * zvec[d] / (2.0 * sigma2 * cp);
    }
    return val;
  };
  double t0 = std::log(std::max(lam2_cur, 1e-12));
  double y  = logf(t0) + std::log(R::unif_rand());
  double w  = 1.0;
  double L  = t0 - w * R::unif_rand(), Rr = L + w;
  for (int k = 0; k < 60 && L  > -40.0 && logf(L)  > y; ++k) L  -= w;
  for (int k = 0; k < 60 && Rr <  40.0 && logf(Rr) > y; ++k) Rr += w;
  if (L  < -40.0) L  = -40.0;
  if (Rr >  40.0) Rr =  40.0;
  for (int k = 0; k < 100; ++k) {
    double t = L + R::unif_rand() * (Rr - L);
    if (logf(t) > y) return std::exp(t);
    if (t < t0) L = t; else Rr = t;
  }
  return lam2_cur;
}

// [[Rcpp::export]]
List cpp_horseshoe(NumericVector X, NumericMatrix Yr, bool intercept,
                   int nburn, int npost, SEXP seed,
                   bool verbose, int printevery, int chain_index) {
  maybe_set_seed(seed);

  IntegerVector dimX = X.attr("dim");
  int D = dimX[0], n = dimX[1], p = dimX[2];

  std::vector<arma::mat> Xd = split_X(X, D, n, p);
  arma::mat Yd(D, n);
  for (int d = 0; d < D; ++d) for (int i = 0; i < n; ++i) Yd(d, i) = Yr(d, i);

  std::vector<arma::mat> XtX(D);
  for (int d = 0; d < D; ++d) XtX[d] = Xd[d].t() * Xd[d];

  bool n_gt_p = (n > p);
  arma::mat beta;  arma::vec alpha;  double sigma2;
  pooled_init(Xd, Yd, intercept, D, n, p, n_gt_p, beta, alpha, sigma2);

  // Data-informed, moderate initialisation of the Horseshoe scales, mirroring
  // Multi-Laplace / Spike-Laplace (which seed their shrinkage variances from the
  // pooled OLS fit and reach full-model R-hat ~1.001). The old raw half-Cauchy
  // starts (tau2 = rcauchy^2, lambda2 = rcauchy^2) are heavy-tailed and leave
  // chains at wildly different scales, stalling cross-chain mixing (R-hat ~1.013).
  // Here we run ONE Gibbs sweep of the eta/lambda2/tau2/kappa block from the OLS
  // beta (kappa = 1, tau2 = 1 seed), using the sampler's own full conditionals:
  // per-chain-random but never extreme. The stationary distribution is unchanged.
  arma::vec beta_mul = beta_mul_cpp(beta, D, p);
  double eta = 1.0, tau2 = 1.0;
  arma::vec kappa(p, arma::fill::ones), lambda2(p);
  for (int j = 0; j < p; ++j)
    lambda2[j] = rinvgamma_cpp((D + 1.0) / 2.0,
                               1.0 / kappa[j] + beta_mul[j] / (2.0 * sigma2 * tau2));
  eta = rinvgamma_cpp(1.0, 1.0 + 1.0 / tau2);
  {
    double bt = 1.0 / eta;
    for (int j = 0; j < p; ++j) bt += beta_mul[j] / lambda2[j] / (2.0 * sigma2);
    tau2 = rinvgamma_cpp((D * p + 1.0) / 2.0, bt);
  }
  for (int j = 0; j < p; ++j) kappa[j] = rinvgamma_cpp(1.0, 1.0 + 1.0 / lambda2[j]);

  arma::mat Xbeta(D, n);
  for (int d = 0; d < D; ++d) Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();

  std::vector<double> dbeta((size_t)npost * D * p, 0.0);
  std::vector<double> dfit((size_t)npost * D * n, 0.0);
  std::vector<double> hmp((size_t)D * n * n, 0.0);
  NumericMatrix post_lambda2(npost, p), post_kappa(npost, p), post_alpha(npost, D);
  NumericVector post_tau2(npost), post_eta(npost), post_sigma2(npost);

  int total = nburn + npost;
  for (int it = 1; it <= total; ++it) {
    progress(verbose, chain_index, it, total, nburn, printevery);
    if (it % 200 == 0) Rcpp::checkUserInterrupt();

    eta = rinvgamma_cpp(1.0, 1.0 + 1.0 / tau2);

    double bt = 1.0 / eta;
    for (int j = 0; j < p; ++j) bt += beta_mul[j] / lambda2[j] / (2.0 * sigma2);
    tau2 = rinvgamma_cpp((D * p + 1.0) / 2.0, bt);

    for (int j = 0; j < p; ++j)
      kappa[j] = rinvgamma_cpp(1.0, 1.0 + 1.0 / lambda2[j]);
    // lambda2[j] is now drawn (marginal, collapsed) in the coordinate scan below.

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
    for (int j = 0; j < p; ++j) SSE_beta += beta_mul[j] / (lambda2[j] * tau2);
    sigma2 = rinvgamma_cpp(D * (n + p) / 2.0, (SSE + SSE_beta) / 2.0);

    // ---- Collapsed coordinate scan: draw lambda2[j] from its MARGINAL conditional
    // (beta_{.,j} integrated out) via slice, then beta_{d,j} | lambda2[j] closed form.
    // Drawing lambda2[j] free of the current beta_{.,j} breaks the lambda-beta funnel;
    // beta follows. phi_j = 1/(lambda2_j tau2) is the beta prior precision. Exact,
    // partially-collapsed Gibbs on the same target as the block sampler.
    std::vector<arma::vec> eres(D);
    for (int d = 0; d < D; ++d) eres[d] = Yd.row(d).t() - alpha[d] - Xbeta.row(d).t();
    for (int j = 0; j < p; ++j) {
      arma::vec cvec(D), zvec(D);
      for (int d = 0; d < D; ++d) {
        double cdj = XtX[d](j, j);
        cvec[d] = cdj;
        zvec[d] = arma::dot(Xd[d].col(j), eres[d]) + cdj * beta(d, j); // X_{d,j}'(resid incl. j)
      }
      lambda2[j] = slice_lam2_hs(lambda2[j], kappa[j], tau2, sigma2, cvec, zvec, D);
      double phi = 1.0 / (lambda2[j] * tau2);
      for (int d = 0; d < D; ++d) {
        double prec = cvec[d] + phi;
        double bnew = R::rnorm(zvec[d] / prec, std::sqrt(sigma2 / prec));
        eres[d] -= Xd[d].col(j) * (bnew - beta(d, j));
        beta(d, j) = bnew;
      }
    }
    for (int d = 0; d < D; ++d) Xbeta.row(d) = (Yd.row(d).t() - alpha[d] - eres[d]).t();
    beta_mul = beta_mul_cpp(beta, D, p);

    // hat matrix (posterior mean, for the four-step projection df) from current scales.
    // Use spdinv_cpp (fork-safe, as in mvn_from_precision), NOT arma::inv_sympd: the
    // latter calls LAPACK directly and segfaults inside a mclapply fork worker on
    // macOS Accelerate (GCD + fork). This runs under nchains>1 & ncores>1.
    if (it > nburn)
      for (int d = 0; d < D; ++d) {
        arma::mat va = spdinv_cpp(XtX[d] + arma::diagmat(1.0 / (lambda2 * tau2)));
        va = 0.5 * (va + va.t());
        arma::mat hp = Xd[d] * va * Xd[d].t();
        for (int a_ = 0; a_ < n; ++a_)
          for (int b_ = 0; b_ < n; ++b_)
            hmp[d + D * a_ + D * n * b_] += hp(a_, b_);
      }

    if (it > nburn) {
      int idx = it - nburn - 1;
      for (int j = 0; j < p; ++j) {
        post_lambda2(idx, j) = lambda2[j];
        post_kappa(idx, j)   = kappa[j];
        for (int d = 0; d < D; ++d)
          dbeta[idx + (size_t)npost * d + (size_t)npost * D * j] = beta(d, j);
      }
      for (int d = 0; d < D; ++d) {
        post_alpha(idx, d) = alpha[d];
        for (int i = 0; i < n; ++i)
          dfit[idx + (size_t)npost * d + (size_t)npost * D * i] =
            Xbeta(d, i) + alpha[d] + R::rnorm(0.0, std::sqrt(sigma2));
      }
      post_tau2[idx] = tau2; post_eta[idx] = eta; post_sigma2[idx] = sigma2;
    }
  }
  for (size_t k = 0; k < hmp.size(); ++k) hmp[k] /= npost;

  List out = finalise(npost, D, p, n, dbeta, dfit, hmp);
  out["post_lambda2"] = post_lambda2;
  out["post_kappa"]   = post_kappa;
  out["post_tau2"]    = post_tau2;
  out["post_eta"]     = post_eta;
  out["post_alpha"]   = post_alpha;
  out["post_sigma2"]  = post_sigma2;
  return out;
}
