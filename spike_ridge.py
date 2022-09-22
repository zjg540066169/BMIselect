#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:00:50 2021

@author: jungang
"""

import pymc3 as pm3
import numpy as np
import theano.tensor as tt
from theano import printing, function
# 1, 2, 4(0.5), 5, 6, 8(0.5), 9, 10(0.45), 11, 12, 13, 14(0.5), 15, 16(0.5), 17(0.5), 18, 20
def spike_ridge(y, X, draw, seed, n_chain, tune, target_accept, p0 = 0.5, v0 = 1, n_thread = 4):
    model = pm3.Model()
    p0 = 0.5#X.shape[0] / X.shape[2]
    v0 = (1 / p0)**2
    with model:
        logsigma2 = pm3.Flat("logsigma2")
        sigma2 = pm3.Deterministic("sigma2", tt.exp(logsigma2))
        g = pm3.Bernoulli("g", p = p0, shape = X.shape[2])
  
        weight = []
        #weight_a0 = []
        for i in range(X.shape[0]):
            weight.append(pm3.Normal("beta"+str(i), mu = 0, sigma = v0, shape = X.shape[2]) * g)
            #weight_a0.append(pm3.Normal("beta0"+str(i), mu = 0, sigma = v0))
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