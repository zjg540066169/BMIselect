#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:39:56 2022

@author: zoujungang
"""

import pymc3 as pm3
import theano.tensor as tt
import numpy as np
from theano import printing, function
from Bayesian_MI_LASSO import Bmi_lasso
import time


class Ridge(Bmi_lasso):
    def __init__(self, y, X, standardize = True):
        super().__init__(y, X, standardize)
        with self.model:
            
            # set prior
            self.logsigma2 = pm3.Flat("logsigma2")
            self.sigma2 = pm3.Deterministic("sigma2", tt.pow(tt.exp(self.logsigma2), 2))
            
            self.logai = pm3.Flat('logai', shape = self.p)
            self.loga0 = pm3.Flat('loga0')
            self.ai = pm3.Deterministic("ai", 1 / tt.exp(self.logai))
            self.a0 = pm3.Deterministic("a0", 1 / tt.exp(self.loga0))
            
            
            # set regression variables
            self.weight = []
            self.weight_a0 = []
            for i in range(self.num_of_imputation):
                self.weight_a0.append(pm3.Normal("alpha"+str(i), mu = 0, sigma = tt.sqrt(self.a0)))
                self.weight.append(pm3.Normal("beta"+str(i), mu = 0, sigma = tt.sqrt(self.ai), shape = self.p))

            # set likelihood
            self.y_obs = []
            for j in range(self.num_of_imputation):
                self.y_obs.append(pm3.Normal("y_obs" + str(j), mu = self.weight_a0[j] + pm3.math.dot(self.X[j], self.weight[j]), sigma = tt.sqrt(self.sigma2), observed = self.y[j]))
                
            
if __name__ == "__main__":
    
    from genDS_MAR import genDS_MAR
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
    miss_index = np.arange(10, 20)
    gendata = genDS_MAR(123, alpha_LM, miss_index, n, p, covmat, beta, sigma) 
    data = gendata["M"]
    X_array = []
    Y_array = []
    for i in range(K):
        X_array.append(data[data.loc[:, "imp"] == i].iloc[:, 1:(p + 1)].to_numpy())
        Y_array.append(data[data.loc[:, "imp"] == i].iloc[:, 0].to_numpy())
    X_array = np.array(X_array)
    Y_array = np.array(Y_array)
    model = Ridge(Y_array, X_array)
    model.sample(10, 10, n_chain = 3)