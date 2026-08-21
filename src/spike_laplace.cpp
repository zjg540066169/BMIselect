// Spike-and-Laplace Bayesian MI-LASSO  (Web Algorithm 4 of supp.tex).
// Partially-collapsed Gibbs: gamma sampled from collapsed model with beta
// integrated out (rank-1 Woodbury / determinant lemma), then beta drawn
// from the uncollapsed conditional.
//
// Loop order replicates the validated reference R sampler:
//   rho -> theta -> lambda2 -> (alpha) -> sigma2 -> Z (collapsed) -> beta.
// hat_matrix_proj uses active-Z columns embedded into the full n x n
// matrix to match the supplement's df expression for Spike-Laplace.
//
// Numerical safeguard for the collapsed log-likelihood: M0 = I + X_0
// diag(lam0) X_0^T is analytically SPD, but floating-point pivots can dip
// below zero when some lambda_j is large.  We retry chol with progressively
// larger diagonal jitter (1e-10 -> 1e-2) before falling back to a
// conservative negative ll, so flips are unbiased under numerical stress.

#include "bmi_utils.h"
using namespace Rcpp;

// DENSE reference (kept for cross-validation): builds the n x n matrix
// M0 = I + X_0 diag(lam0) X_0^T and factorizes it.  Sum over imputations of the
// collapsed log-likelihood difference for flipping gamma_j.  Uses raw Y (no
// alpha).  Returns sum_d (ll_plus - ll_minus).
static double sl_loglik_diff_dense(const std::vector<arma::mat>& Xd,
                             const arma::mat& Yd,
                             const arma::vec& lambda2,
                             double sigma2,
                             const arma::ivec& Z, int j, int D, int n) {
  double total = 0.0;
  for (int d = 0; d < D; ++d) {
    std::vector<int> sel_other;
    for (int k = 0; k < (int)Z.n_elem; ++k)
      if (Z[k] == 1 && k != j) sel_other.push_back(k);

    arma::mat M0 = arma::eye(n, n);
    if (!sel_other.empty()) {
      arma::mat X0(n, sel_other.size());
      arma::vec lam0(sel_other.size());
      for (size_t c = 0; c < sel_other.size(); ++c) {
        X0.col(c) = Xd[d].col(sel_other[c]);
        lam0[c]   = lambda2[sel_other[c]];
      }
      M0 += X0 * arma::diagmat(lam0) * X0.t();
    }
    arma::mat U0;
    if (!arma::chol(U0, M0)) {
      double jit = 1e-10 * std::max(1.0, arma::trace(M0) / (double)n);
      bool fixed = false;
      for (int tries = 0; tries < 8; ++tries) {
        if (arma::chol(U0, M0 + jit * arma::eye(n, n))) { fixed = true; break; }
        jit *= 10.0;
      }
      if (!fixed) { total += -1e10; continue; }
    }
    double logdet0 = 2.0 * arma::accu(arma::log(U0.diag()));

    arma::vec y = Yd.row(d).t();
    arma::vec zY = arma::solve(arma::trimatl(U0.t()), y);
    arma::vec wY = arma::solve(arma::trimatu(U0), zY);
    double q0 = arma::dot(y, wY);

    arma::vec xj = Xd[d].col(j);
    arma::vec zx = arma::solve(arma::trimatl(U0.t()), xj);
    arma::vec vj = arma::solve(arma::trimatu(U0), zx);
    double s0 = arma::dot(xj, vj);
    double t0 = arma::dot(y, vj);
    double lamj = lambda2[j];

    double alpha = 1.0 + lamj * s0;
    double logdet_plus = logdet0 + std::log(alpha);
    double q_plus = q0 - (lamj * t0 * t0) / alpha;

    double ll_minus = -0.5 * (n * std::log(sigma2) + logdet0 + q0 / sigma2);
    double ll_plus  = -0.5 * (n * std::log(sigma2) + logdet_plus + q_plus / sigma2);
    total += (ll_plus - ll_minus);
  }
  return total;
}

// WOODBURY version: identical collapsed log-likelihood difference, but the
// n x n matrix M0 = I + X_0 Lambda_0 X_0^T is replaced by the |s0| x |s0| inner
// matrix B = Lambda_0^{-1} + X_0^T X_0 (matrix determinant lemma + Woodbury):
//   logdet(M0) = sum_k log(lambda0_k) + logdet(B)
//   u^T M0^{-1} v = u^T v - (X_0^T u)^T B^{-1} (X_0^T v).
// G[d] = X_d^T X_d (p x p), Xty[d] = X_d^T y_d, yty[d] = y_d^T y_d are constant
// across iterations and precomputed once, so each evaluation is O(|s0|^3).
static double sl_loglik_diff(const std::vector<arma::mat>& G,
                             const std::vector<arma::vec>& Xty,
                             const arma::vec& yty,
                             const arma::vec& lambda2,
                             double sigma2,
                             const arma::ivec& Z, int j, int D, int n) {
  double total = 0.0;
  for (int d = 0; d < D; ++d) {
    std::vector<arma::uword> s0v;
    for (arma::uword k = 0; k < Z.n_elem; ++k)
      if (Z[k] == 1 && (int)k != j) s0v.push_back(k);

    double logdet0, q0, sq, t0;            // for the -j (base) model M0
    double xjxj = G[d](j, j);              // xj^T xj
    double yxj  = Xty[d](j);               // y^T xj
    if (s0v.empty()) {                     // M0 = I_n
      logdet0 = 0.0; q0 = yty[d]; sq = xjxj; t0 = yxj;
    } else {
      arma::uvec idx = arma::conv_to<arma::uvec>::from(s0v);
      arma::uword m = idx.n_elem;
      arma::mat B = G[d].submat(idx, idx);                 // X0^T X0
      for (arma::uword c = 0; c < m; ++c) B(c, c) += 1.0 / lambda2[idx[c]]; // + Lambda0^{-1}
      arma::mat R;
      if (!arma::chol(R, B)) {                              // B = R^T R, R upper
        double jit = 1e-10 * std::max(1.0, arma::trace(B) / (double)m);
        bool fixed = false;
        for (int tries = 0; tries < 8; ++tries) {
          if (arma::chol(R, B + jit * arma::eye(m, m))) { fixed = true; break; }
          jit *= 10.0;
        }
        if (!fixed) { total += -1e10; continue; }
      }
      double logdetB = 2.0 * arma::accu(arma::log(R.diag()));
      double sumlog_lam = 0.0;
      for (arma::uword c = 0; c < m; ++c) sumlog_lam += std::log(lambda2[idx[c]]);
      logdet0 = sumlog_lam + logdetB;                      // det lemma

      arma::vec Xty0 = Xty[d].elem(idx);                   // X0^T y   (m)
      arma::vec gcolj = G[d].col(j);                       // materialize subview
      arma::vec Xtxj = gcolj.elem(idx);                    // X0^T xj  (m)
      // a = B^{-1} rhs  via  R^T z = rhs ;  R a = z
      arma::vec ay = arma::solve(arma::trimatu(R), arma::solve(arma::trimatl(R.t()), Xty0));
      arma::vec ax = arma::solve(arma::trimatu(R), arma::solve(arma::trimatl(R.t()), Xtxj));
      q0 = yty[d] - arma::dot(Xty0, ay);                   // y^T M0^{-1} y
      sq = xjxj   - arma::dot(Xtxj, ax);                   // xj^T M0^{-1} xj
      t0 = yxj    - arma::dot(Xty0, ax);                   // y^T M0^{-1} xj
    }
    double lamj = lambda2[j];
    double alpha = 1.0 + lamj * sq;                        // rank-1 add of column j
    double logdet_plus = logdet0 + std::log(alpha);
    double q_plus = q0 - (lamj * t0 * t0) / alpha;
    double ll_minus = -0.5 * (n * std::log(sigma2) + logdet0 + q0 / sigma2);
    double ll_plus  = -0.5 * (n * std::log(sigma2) + logdet_plus + q_plus / sigma2);
    total += (ll_plus - ll_minus);
  }
  return total;
}

// Validation export: returns a p x 2 matrix of (dense, woodbury) gamma_j
// log-likelihood differences for a given (lambda2, sigma2, Z), to confirm the
// two implementations agree to machine precision.
// [[Rcpp::export]]
NumericMatrix cpp_sl_loglik_check(NumericVector X, NumericMatrix Yr,
                                  NumericVector lambda2_, double sigma2,
                                  IntegerVector Z_) {
  IntegerVector dimX = X.attr("dim");
  int D = dimX[0], n = dimX[1], p = dimX[2];
  std::vector<arma::mat> Xd = split_X(X, D, n, p);
  arma::mat Yd(D, n);
  for (int d = 0; d < D; ++d) for (int i = 0; i < n; ++i) Yd(d, i) = Yr(d, i);
  arma::vec lambda2(p); for (int j = 0; j < p; ++j) lambda2[j] = lambda2_[j];
  arma::ivec Z(p);      for (int j = 0; j < p; ++j) Z[j] = Z_[j];
  std::vector<arma::mat> G(D); std::vector<arma::vec> Xty(D); arma::vec yty(D);
  for (int d = 0; d < D; ++d) {
    G[d] = Xd[d].t() * Xd[d];
    arma::vec yv = Yd.row(d).t();
    Xty[d] = Xd[d].t() * yv;
    yty[d] = arma::dot(yv, yv);
  }
  NumericMatrix out(p, 2);
  for (int j = 0; j < p; ++j) {
    out(j, 0) = sl_loglik_diff_dense(Xd, Yd, lambda2, sigma2, Z, j, D, n);
    out(j, 1) = sl_loglik_diff(G, Xty, yty, lambda2, sigma2, Z, j, D, n);
  }
  return out;
}

// [[Rcpp::export]]
List cpp_spike_laplace(NumericVector X, NumericMatrix Yr, bool intercept,
                       double a, double b, int nburn, int npost,
                       SEXP seed, bool verbose,
                       int printevery, int chain_index) {
  maybe_set_seed(seed);

  IntegerVector dimX = X.attr("dim");
  int D = dimX[0], n = dimX[1], p = dimX[2];
  if (R_IsNA(b)) b = (D + 1.0) / (2.0 * D) / (a - 1.0);

  std::vector<arma::mat> Xd = split_X(X, D, n, p);
  arma::mat Yd(D, n);
  for (int d = 0; d < D; ++d) for (int i = 0; i < n; ++i) Yd(d, i) = Yr(d, i);

  // Gram quantities for the collapsed gamma update: constant across iterations,
  // computed once.  G[d] = X_d^T X_d, Xty[d] = X_d^T y_d, yty[d] = y_d^T y_d.
  std::vector<arma::mat> Gram(D); std::vector<arma::vec> Xty(D); arma::vec yty(D);
  for (int d = 0; d < D; ++d) {
    Gram[d] = Xd[d].t() * Xd[d];
    arma::vec yv = Yd.row(d).t();
    Xty[d] = Xd[d].t() * yv;
    yty[d] = arma::dot(yv, yv);
  }

  bool n_gt_p = (n > p);
  arma::mat beta;  arma::vec alpha;  double sigma2;
  pooled_init(Xd, Yd, intercept, D, n, p, n_gt_p, beta, alpha, sigma2);

  arma::vec theta(p);
  arma::ivec Z(p);
  for (int j = 0; j < p; ++j) {
    theta[j] = R::rbeta(a, b);
    Z[j] = (R::unif_rand() < theta[j]) ? 1 : 0;
  }
  double rho = R::rgamma(a, b);
  arma::vec lambda2(p);
  for (int j = 0; j < p; ++j) lambda2[j] = R::rgamma((D + 1.0) / 2.0, 2.0 / (D * rho));

  arma::mat Xbeta(D, n);
  for (int d = 0; d < D; ++d) Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();
  arma::vec beta_mul = beta_mul_cpp(beta, D, p);

  std::vector<double> dbeta((size_t)npost * D * p, 0.0);
  std::vector<double> dfit((size_t)npost * D * n, 0.0);
  std::vector<double> hmp((size_t)D * n * n, 0.0);
  NumericMatrix post_gamma(npost, p), post_theta(npost, p),
                post_lambda2(npost, p), post_alpha(npost, D);
  NumericVector post_rho(npost), post_sigma2(npost);

  int total = nburn + npost;
  for (int it = 1; it <= total; ++it) {
    progress(verbose, chain_index, it, total, nburn, printevery);
    if (it % 50 == 0) Rcpp::checkUserInterrupt();

    // rho | lambda2 (previous)
    double rate = 1.0 / b + D * arma::accu(lambda2) / 2.0;
    rho = R::rgamma(a + p * (D + 1.0) / 2.0, 1.0 / rate);

    // theta | Z (previous)
    for (int j = 0; j < p; ++j)
      theta[j] = R::rbeta(1.0 + Z[j], 2.0 - Z[j]);

    // lambda2 | beta_mul (previous), rho, sigma2, Z
    for (int j = 0; j < p; ++j) {
      if (Z[j] == 1) lambda2[j] = rgig_half_cpp(beta_mul[j] / sigma2, D * rho);
      else           lambda2[j] = R::rgamma((D + 1.0) / 2.0, 2.0 / (D * rho));
    }

    if (intercept)
      for (int d = 0; d < D; ++d) {
        double mu = arma::mean(Yd.row(d).t() - Xbeta.row(d).t());
        alpha[d] = R::rnorm(mu, std::sqrt(sigma2 / n));
      }

    // sigma2 | Xbeta (prev beta), beta_mul (prev), Z, lambda2 (new)
    double SSE = 0.0;
    for (int d = 0; d < D; ++d) {
      arma::vec res = Yd.row(d).t() - Xbeta.row(d).t() - alpha[d];
      SSE += arma::dot(res, res);
    }
    // Only the active (Z_j = 1) coefficients carry a sigma2-scaled slab prior
    // N(0, sigma2 * lambda2_j), so only they contribute D/2 each to the shape.
    // The spike coefficients are exactly 0 and add nothing: shape is D(n+|Z|)/2,
    // not D(n+p)/2 (which would count the null dimensions the rate already excludes).
    double SSE_beta = 0.0;
    int p_active = 0;
    for (int j = 0; j < p; ++j)
      if (Z[j] == 1) { SSE_beta += beta_mul[j] / lambda2[j]; ++p_active; }
    sigma2 = rinvgamma_cpp(D * (n + p_active) / 2.0, (SSE + SSE_beta) / 2.0);

    // Z | . (collapsed, sequential) -- Woodbury collapsed likelihood
    for (int j = 0; j < p; ++j) {
      double Rj = sl_loglik_diff(Gram, Xty, yty, lambda2, sigma2, Z, j, D, n);
      double pr = theta[j] / (theta[j] + (1.0 - theta[j]) * std::exp(-Rj));
      Z[j] = (R::unif_rand() < pr) ? 1 : 0;
    }

    // beta | Z
    beta.zeros(D, p);
    arma::mat hp_d(n, n);
    std::vector<int> sel;
    for (int j = 0; j < p; ++j) if (Z[j] == 1) sel.push_back(j);
    if (!sel.empty()) {
      arma::vec invl(sel.size());
      for (size_t c = 0; c < sel.size(); ++c) invl[c] = 1.0 / lambda2[sel[c]];
      arma::mat Prec = arma::diagmat(invl);
      for (int d = 0; d < D; ++d) {
        arma::mat Xs(n, sel.size());
        for (size_t c = 0; c < sel.size(); ++c) Xs.col(c) = Xd[d].col(sel[c]);
        arma::mat XtXs = Xs.t() * Xs;
        arma::vec r = Yd.row(d).t() - alpha[d];
        arma::vec bs = mvn_from_precision(Xs, XtXs, Prec, r, sigma2, hp_d);
        for (size_t c = 0; c < sel.size(); ++c) beta(d, sel[c]) = bs[c];
        if (it > nburn)
          for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y)
              hmp[d + D * x + D * n * y] += hp_d(x, y);
      }
    }
    for (int d = 0; d < D; ++d) Xbeta.row(d) = (Xd[d] * beta.row(d).t()).t();
    beta_mul = beta_mul_cpp(beta, D, p);

    if (it > nburn) {
      int idx = it - nburn - 1;
      for (int j = 0; j < p; ++j) {
        post_gamma(idx, j)   = Z[j];
        post_theta(idx, j)   = theta[j];
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
      post_rho[idx] = rho; post_sigma2[idx] = sigma2;
    }
  }
  for (size_t k = 0; k < hmp.size(); ++k) hmp[k] /= npost;

  List out = finalise(npost, D, p, n, dbeta, dfit, hmp);
  out["post_rho"]     = post_rho;
  out["post_gamma"]   = post_gamma;
  out["post_theta"]   = post_theta;
  out["post_alpha"]   = post_alpha;
  out["post_lambda2"] = post_lambda2;
  out["post_sigma2"]  = post_sigma2;
  out["a"] = a; out["b"] = b;
  return out;
}
