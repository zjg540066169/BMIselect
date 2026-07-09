// Horseshoe Bayesian MI-LASSO  (Web Algorithm 2 of supp.tex).
// Auxiliary inverse-gamma representation of half-Cauchy (Makalic & Schmidt 2016).
// Loop order: eta -> tau2 -> kappa -> lambda2 -> (alpha) -> sigma2 -> beta.

#include "bmi_utils.h"
using namespace Rcpp;

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

  double tau2 = std::pow(R::rcauchy(0.0, 1.0), 2.0);
  arma::vec lambda2(p);
  for (int j = 0; j < p; ++j) lambda2[j] = std::pow(R::rcauchy(0.0, 1.0), 2.0);
  double eta = rinvgamma_cpp(1.0, 1.0 + 1.0 / tau2);
  arma::vec kappa(p);
  for (int j = 0; j < p; ++j)
    kappa[j] = rinvgamma_cpp(1.0, 1.0 + 1.0 / lambda2[j]);

  arma::mat Xbeta(D, n);
  for (int d = 0; d < D; ++d) Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();
  arma::vec beta_mul = beta_mul_cpp(beta, D, p);

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
    for (int j = 0; j < p; ++j)
      lambda2[j] = rinvgamma_cpp((D + 1.0) / 2.0,
                                 1.0 / kappa[j] + beta_mul[j] / (2.0 * sigma2 * tau2));

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

    arma::mat Prec = arma::diagmat(1.0 / (lambda2 * tau2));
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
