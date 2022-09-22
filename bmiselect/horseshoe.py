#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:43:04 2021

@author: jungang
"""
from genDS_MAR import genDS_MAR
from MI_lasso import MI_lasso
from pmi_l2 import pmi_l2
import pymc3 as pm3
import theano.tensor as tt
import numpy as np
from theano import printing, function
# 2,5, 6,11, 13, 15
def horseshoe(y, X, draw, seed, n_chain, tune, target_accept, n_thread = 4, normalized = True):
    model = pm3.Model()
    
    with model:
        logsigma2 = pm3.Flat("logsigma2")
        sigma2 = pm3.Deterministic("sigma2", tt.exp(logsigma2))
        
        tau = pm3.HalfCauchy("tau", beta = 1)
        
        
        
 
        lambda_ = pm3.HalfCauchy("lambda", beta = 1, shape = X.shape[2])
        
        if normalized == False:
            lambda_0 = pm3.HalfCauchy("lambda0", beta = 1)
        
  
        weight = []
        if normalized == False:
            weight_a0 = []
        for i in range(X.shape[0]):
            if normalized == False:
                weight_a0.append(pm3.Normal("beta0"+str(i), mu = 0, sigma = tt.pow(lambda_0, 2) * tt.pow(tau, 2)))
            
            weight.append(pm3.Normal("beta"+str(i), mu = 0, sigma = tt.pow(lambda_, 2) * tt.pow(tau, 2), shape = X.shape[2]))
            
        #beta = pm3.MvNormal("beta" , mu = np.zeros(X.shape[0]), cov = tt.diag(ai))
        y_obs = []
        for j in range(X.shape[0]):
            if normalized == False:
                y_obs.append(pm3.Normal("y_obs" + str(j), mu = weight_a0[j] + pm3.math.dot(X[j], weight[j]), sigma = sigma2, observed = y[j]))
            else:
                y_obs.append(pm3.Normal("y_obs" + str(j), mu = pm3.math.dot(X[j], weight[j]), sigma = sigma2, observed = y[j]))
    with model:
        try:
            trace3 = pm3.sample(draws = draw, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = tune, target_accept = target_accept, max_treedepth = 10)
        except ValueError:
            trace3 = pm3.sample(draws = draw, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = tune)
    #status = pm3.traceplot(trace3)
    
    #beta, beta0 = bayesian_mixture(trace3, X.shape[0])
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
    
    
    
  
    draw = 1000
    tune = 2000
    
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
    for i in range(K):
        X_array.append(data[data.loc[:, "imp"] == i].iloc[:, 1:(p + 1)].to_numpy())
        Y_array.append(data[data.loc[:, "imp"] == i].iloc[:, 0].to_numpy())
    X_array = np.array(X_array)
    Y_array = np.array(Y_array)
    
    beta = horseshoe(Y_array, X_array, draw, 123, n_chain, tune, target_accept, n_thread = 4)
