#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:09:03 2021

@author: jungang
"""

import pymc3 as pm3
import theano.tensor as tt
from theano import printing, function
import numpy as np
# 1,1,1,1;   2,5(0.4),6(0.33),7(0.08),11,12,15(0.09),20(0.4)
# 0.002,1,1,1;     2,5,6,11,13
# 0.5,1,1,1;     2,5,6,11,12(0.5), 13(0.5), 15(0.5)
    
    
def spike_laplace(y, X, draw, seed, n_chain, tune, target_accept, lambda_ = 6/11, a = 1, b = 1, n_thread = 4):
    model = pm3.Model()
    with model:
        logsigma2 = pm3.Flat("logsigma2")
        sigma2 = pm3.Deterministic("sigma2", tt.exp(logsigma2))
        p = pm3.Beta("p", alpha = a, beta = b, shape = X.shape[2])
        tg = pm3.Gamma("tg", alpha = (X.shape[0] + 1) / 2, beta = 2/(X.shape[0] * lambda_), shape = X.shape[2])
        g = pm3.Bernoulli("g", p = p, shape = X.shape[2])
        weight = []
# =============================================================================
#         tg0 = pm3.Gamma("tg0", alpha = (X.shape[0] + 1) / 2, beta = tt.pow(lambda_, 2) / 2)
#         
#         weight_a0 = []
# =============================================================================
        for i in range(X.shape[0]):
            weight.append(pm3.Normal("beta"+str(i), mu = 0, sigma = tg, shape = X.shape[2]) * g)
            #weight_a0.append(pm3.Normal("beta0"+str(i), mu = 0, sigma = sigma2 * tg0))
        #beta = pm3.MvNormal("beta" , mu = np.zeros(X.shape[0]), cov = tt.diag(ai))
        y_obs = []
        for j in range(X.shape[0]):
            #y_obs.append(pm3.Normal("y_obs" + str(j), mu = weight_a0[j] + pm3.math.dot(X[j], weight[j]), sigma = sigma2, observed = y[j]))
            y_obs.append(pm3.Normal("y_obs" + str(j), mu = pm3.math.dot(X[j], weight[j]), sigma = sigma2, observed = y[j]))
    with model:
        try:
            trace3 = pm3.sample(draws = draw, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = tune, target_accept = target_accept)
        except ValueError:
            trace3 = pm3.sample(draws = draw, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = tune)
    #status = pm3.traceplot(trace3)
    
    #beta, beta0 = bayesian_mixture(trace3, X.shape[0])
    beta = bayesian_mixture(trace3, X.shape[0])
    return beta, trace3["g"]


def bayesian_mixture(trace, num_impute):
    beta = []
    #beta0 = []
    for i in range(num_impute):
        beta.append(trace["beta" + str(i)])
        #beta0.append(trace["beta0" + str(i)])
    beta = np.concatenate(beta, axis=0)
    #beta0 = np.concatenate(beta0, axis=0)
    return beta#, beta0 

def EM_lambda(trace, X):
    return np.sqrt((X.shape[0] + X.shape[2] + 1) / (trace["tg"].mean(0).sum() + trace["tg0"].mean()))


def EM_lambda_group(trace, X):
    lambda_ = np.zeros(X.shape[2])
    for i in range(X.shape[2]):
        lambda_[i] = np.sqrt((1 + X.shape[2]) / (trace["tg"].mean(0)[i] * X.shape[0]))
    return lambda_
    
    
    