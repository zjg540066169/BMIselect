## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

* The check reports the URL
  https://www.columbia.edu/~qc2138/Downloads/software/MI-lasso.R (in the
  vignette) as possibly invalid with status 403. This link is valid and opens
  in a web browser; the host (Columbia University personal web pages) returns
  403 to automated, non-interactive requests. It is the original source of the
  MI_LASSO() implementation, which we credit at that location.

Here are some acronyms:
* Zou, Wang, Chen: Authors' last names.
* ARD: Automatic Relevance Determination. Name of a prior.
* Horseshoe: Name of a prior.
* Multi_Laplace: Name of a prior.
* Spike_Laplace: Name of a prior.
* MI: Multiple imputation.
* LASSO: Least absolute shrinkage and selection operator.
* MI-LASSO: multiple-imputation least absolute shrinkage and selection operator.
* MCMC: Monte-Carlo Markov Chain.
* BIC: Bayesian information criterion.
* DF: degree of freedom.
* SNC: scaled neighborhood criterion.
* MCAR: Missing completely at random.
* MAR: Missing at random.


Also, in the description text, the models' names (Multi-Laplace, Horseshoe, ARD, 
Spike-Laplace) are names of models in the paper, not the functions.
