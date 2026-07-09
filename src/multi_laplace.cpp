// Multi-Laplace Bayesian MI-LASSO  (Web Algorithm 1 of supp.tex).
//
// Loop order replicates the reference R sampler:
//   rho  ->  lambda2  ->  (alpha)  ->  sigma2  ->  beta.
// Post-burn-in accumulation is on iter > nburn (hat_matrix_proj / npost
// matches the (1/T) sum_{t=1}^T estimator in supplement eq (7)).

#include "bmi_utils.h"
using namespace Rcpp;

// [[Rcpp::export]]
List cpp_multi_laplace(NumericVector X, NumericMatrix Yr, bool intercept,
                       double h, double v, int nburn, int npost,
                       SEXP seed, bool verbose,
                       int printevery, int chain_index) {
  maybe_set_seed(seed);

  IntegerVector dimX = X.attr("dim");
  int D = dimX[0], n = dimX[1], p = dimX[2];
  if (R_IsNA(v)) v = (D + 1.0) / D / (h - 1.0);

  std::vector<arma::mat> Xd = split_X(X, D, n, p);
  arma::mat Yd(D, n);
  for (int d = 0; d < D; ++d) for (int i = 0; i < n; ++i) Yd(d, i) = Yr(d, i);

  std::vector<arma::mat> XtX(D);
  for (int d = 0; d < D; ++d) XtX[d] = Xd[d].t() * Xd[d];

  bool n_gt_p = (n > p);
  arma::mat beta;  arma::vec alpha;  double sigma2;
  pooled_init(Xd, Yd, intercept, D, n, p, n_gt_p, beta, alpha, sigma2);

  double rho = R::rgamma(h, v);
  arma::vec beta_mul = beta_mul_cpp(beta, D, p);
  arma::vec lambda2(p);
  for (int j = 0; j < p; ++j)
    lambda2[j] = rgig_half_cpp(beta_mul[j] / sigma2, D * rho);

  arma::mat Xbeta(D, n);
  for (int d = 0; d < D; ++d) Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();

  std::vector<double> dbeta((size_t)npost * D * p, 0.0);
  std::vector<double> dfit((size_t)npost * D * n, 0.0);
  std::vector<double> hmp((size_t)D * n * n, 0.0);
  NumericMatrix post_lambda2(npost, p);
  NumericMatrix post_alpha(npost, D);
  NumericVector post_rho(npost), post_sigma2(npost);

  int total = nburn + npost;
  for (int it = 1; it <= total; ++it) {
    progress(verbose, chain_index, it, total, nburn, printevery);
    if (it % 200 == 0) Rcpp::checkUserInterrupt();

    // rho | lambda2
    double rate_rho = 1.0 / v + D * arma::accu(lambda2) / 2.0;
    rho = R::rgamma(h + p * (D + 1.0) / 2.0, 1.0 / rate_rho);

    // lambda2 | beta_mul, rho, sigma2
    for (int j = 0; j < p; ++j)
      lambda2[j] = rgig_half_cpp(beta_mul[j] / sigma2, D * rho);

    // alpha | .
    if (intercept)
      for (int d = 0; d < D; ++d) {
        double mu = arma::mean(Yd.row(d).t() - Xbeta.row(d).t());
        alpha[d] = R::rnorm(mu, std::sqrt(sigma2 / n));
      }

    // sigma2 | .
    double SSE = 0.0;
    for (int d = 0; d < D; ++d) {
      arma::vec res = Yd.row(d).t() - Xbeta.row(d).t() - alpha[d];
      SSE += arma::dot(res, res);
    }
    double SSE_beta = 0.0;
    for (int j = 0; j < p; ++j) SSE_beta += beta_mul[j] / lambda2[j];
    sigma2 = rinvgamma_cpp(D * (n + p) / 2.0, (SSE + SSE_beta) / 2.0);

    // beta | .
    arma::mat Prec = arma::diagmat(1.0 / lambda2);
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
        for (int d = 0; d < D; ++d)
          dbeta[idx + (size_t)npost * d + (size_t)npost * D * j] = beta(d, j);
      }
      for (int d = 0; d < D; ++d) {
        post_alpha(idx, d) = alpha[d];
        for (int i = 0; i < n; ++i)
          dfit[idx + (size_t)npost * d + (size_t)npost * D * i] =
            Xbeta(d, i) + alpha[d] + R::rnorm(0.0, std::sqrt(sigma2));
      }
      post_rho[idx] = rho;
      post_sigma2[idx] = sigma2;
    }
  }
  for (size_t k = 0; k < hmp.size(); ++k) hmp[k] /= npost;

  List out = finalise(npost, D, p, n, dbeta, dfit, hmp);
  out["post_lambda2"] = post_lambda2;
  out["post_alpha"]   = post_alpha;
  out["post_rho"]     = post_rho;
  out["post_sigma2"]  = post_sigma2;
  out["h"] = h; out["v"] = v;
  return out;
}
