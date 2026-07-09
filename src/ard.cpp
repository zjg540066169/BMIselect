// ARD Bayesian MI-LASSO  (Web Algorithm 3 of supp.tex).
// Loop order: (alpha) -> sigma2 -> psi2 -> beta.
// psi2_j is capped at 1/eps (eps = 1e-6) to prevent over/underflow when
// beta_mul_j is tiny -- matches the safeguard in the reference R sampler.

#include "bmi_utils.h"
using namespace Rcpp;

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

    for (int j = 0; j < p; ++j) {
      psi2[j] = R::rgamma(D / 2.0, 2.0 * sigma2 / beta_mul[j]);
      psi2[j] = std::min(1.0 / eps, psi2[j]);
    }

    arma::mat Prec = arma::diagmat(psi2);
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
    beta_mul = beta_mul_cpp(beta, D, p);

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
