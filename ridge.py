#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:23:22 2021

@author: jungang
"""
from genDS_MAR import genDS_MAR
from MI_lasso import MI_lasso
from pmi_l2 import pmi_l2
import pymc3 as pm3
import numpy as np
import theano.tensor as tt
from theano import printing, function

import logging
logger = logging.getLogger("pymc3")
logger.propagate = False

# alpha=1, beta = 1;   select:  2,5,6,11

def ridge(y, X, draw, seed, n_chain, tune, target_accept, n_thread = 4):
    model = pm3.Model()
    
    with model:
        logsigma2 = pm3.Flat("logsigma2")
        sigma2 = pm3.Deterministic("sigma2", tt.exp(logsigma2))
        
        #print(f(sigma2))
        logai = pm3.Flat('logai', shape = X.shape[2])
        #loga0 = pm3.Flat('loga0')
        #print(f(logai))
        ai = pm3.Deterministic("ai", 1 / tt.exp(logai))
        #a0 = pm3.Deterministic("a0", 1 / tt.exp(loga0))
        weight = []
        #weight_a0 = []
        for i in range(X.shape[0]):
            weight.append(pm3.Normal("beta"+str(i), mu = 0, sigma = ai, shape = X.shape[2]))
            #weight_a0.append(pm3.Normal("beta0"+str(i), mu = 0, sigma = a0))
        #beta = pm3.MvNormal("beta" , mu = np.zeros(X.shape[0]), cov = tt.diag(ai))
        y_obs = []
        for j in range(X.shape[0]):
           # y_obs.append(pm3.Normal("y_obs" + str(j), mu = weight_a0[j] + pm3.math.dot(X[j], weight[j]), sigma = sigma2, observed = y[j]))
           y_obs.append(pm3.Normal("y_obs" + str(j), mu = pm3.math.dot(X[j], weight[j]), sigma = sigma2, observed = y[j]))
            
    with model:
        try:
            trace3 = pm3.sample(draws = draw, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = tune, target_accept = target_accept)
        except ValueError:
            trace3 = pm3.sample(draws = draw, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = tune)
    #status = pm3.traceplot(trace3)
    
    #beta, beta0 = bayesian_mixture(trace3, X.shape[0])
    #return beta, beta0
    beta = bayesian_mixture(trace3, X.shape[0])
    return beta


def bayesian_mixture(trace, num_impute):
    beta = []
    #beta0 = []
    for i in range(num_impute):
        beta.append(trace["beta" + str(i)])
        #beta0.append(trace["beta0" + str(i)])
    beta = np.concatenate(beta, axis=0)
    #beta0 = np.concatenate(beta0, axis=0)
    return beta#, beta0


if __name__ == "__main__":
    
    
    n = 100
    p = 20
    K = 5
    W = np.diag(np.repeat(1, n))
    rho = 0.1
    SNR = 3
    important = np.array([1,2,3,11,12,13]) - 1
    noise = np.concatenate([np.arange(3, 10),np.arange(13, 20) ])
    truth = np.array([False for i in range(20)])
    truth[important] = True
    
    
    
  
    draw = 5
    tune = 5
    
    n_chain = 4
    n_thread = 4
    target_accept = 0.9
    eps = 1e-2
    
    # compound symmetric correlation matrix with off-diagonal value of "corr"
    covmat = np.ones((p,p)) * rho
    covmat[np.diag_indices_from(covmat)] = 1
    
    beta = np.array([0.5,1,2,0,0,0,0,0,0,0,0.5,1,2, 0,0,0,0,0,0,0])
    sigma = np.sqrt(beta.dot(covmat).dot(beta)/SNR) 
    true_index = np.array([1, 1, 1, 0,0,0,0,0,0,0,1, 1, 1, 0,0,0,0,0,0,0])
    
    alpha_LM = np.array([
    -4,0.5,0,0,0,0,0,0,0,0,0,0.5,
    -4,0,0.5,0,0,0,0,0,0,0,0,0.5,
    -4,0,0,0.5,0,0,0,0,0,0,0,0.5,
    -4,0,0,0,0.5,0,0,0,0,0,0,0.5,
    -4,0,0,0,0,0.5,0,0,0,0,0,0.5,
    -4,0,0,0,0,0,0.5,0,0,0,0,0.5,
    -4,0,0,0,0,0,0,0.5,0,0,0,0.5,
    -4,0,0,0,0,0,0,0,0.5,0,0,0.5,
    -4,0,0,0,0,0,0,0,0,0.5,0,0.5,
    -4,0,0,0,0,0,0,0,0,0,0.5,0.5
    ]).reshape(10, 12)
    
    alpha_HM = np.array([
    -1.8,0.5,0,0,0,0,0,0,0,0,0,0.5,
    -1.8,0,0.5,0,0,0,0,0,0,0,0,0.5,
    -1.8,0,0,0.5,0,0,0,0,0,0,0,0.5,
    -1.8,0,0,0,0.5,0,0,0,0,0,0,0.5,
    -1.8,0,0,0,0,0.5,0,0,0,0,0,0.5,
    -1.8,0,0,0,0,0,0.5,0,0,0,0,0.5,
    -1.8,0,0,0,0,0,0,0.5,0,0,0,0.5,
    -1.8,0,0,0,0,0,0,0,0.5,0,0,0.5,
    -1.8,0,0,0,0,0,0,0,0,0.5,0,0.5,
    -1.8,0,0,0,0,0,0,0,0,0,0.5,0.5
    ]).reshape(10, 12)
    gendata = genDS_MAR(123, alpha_LM, n, p, covmat, beta, sigma) 
    data = gendata["M"]
    X_array = []
    Y_array = []
    X_mean = []
    X_std = []
    Y_mean = []
    for i in range(K):
        X_array.append(data[data.loc[:, "imp"] == i].iloc[:, 1:(p + 1)].to_numpy())
        Y_array.append(data[data.loc[:, "imp"] == i].iloc[:, 0].to_numpy())
        X_mean.append(X_array[i].mean(0))
        X_std.append(X_array[i].std(0))
        Y_mean.append(Y_array[i].mean(0))
        
        X_array[i] = (X_array[i] - X_mean[i]) / X_std[i]
        Y_array[i] = Y_array[i] - Y_mean[i]
    X_array = np.array(X_array)
    Y_array = np.array(Y_array)

    beta = ridge(Y_array, X_array, draw, 123, n_chain, tune, target_accept, alpha = 1, beta = 1, n_thread = 4)
