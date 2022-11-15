#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:02:23 2022

This class provides the model of Spike-Laplace model

Three hyperparameters: lambda, a, b

@author: Jungang Zou
"""

import pymc3 as pm3
import theano.tensor as tt
import numpy as np
from theano import printing, function
from bmiselect.models.Bayesian_MI_LASSO import Bmi_lasso
import time


class Spike_laplace(Bmi_lasso):
    def __init__(self, y, X, standardize = True, lambda_ = 6/11, a = 1, b = 1):
        super().__init__(y, X, standardize)
        if lambda_ <= 0 or a <= 0 or b <= 0:
            raise Exception("All hyper-parameters should be positive")
        self.type_model = "discrete"
        with self.model:
            # set parameter
            self.lambda_ = lambda_
            self.a = a
            self.b = b
            
            # set prior
            self.logsigma2 = pm3.Flat("logsigma2")
            self.sigma2 = pm3.Deterministic("sigma2", tt.pow(tt.exp(self.logsigma2), 2))
            self.pi = pm3.Beta("g", alpha = self.a, beta = self.b, shape = self.p)
            self.tg = pm3.Gamma("tau", alpha = (self.num_of_imputation + 1) / 2, beta = 2/(self.num_of_imputation * self.lambda_), shape = self.p)
            self.tg0 = pm3.Gamma("tau0", alpha = (self.num_of_imputation + 1) / 2, beta = 2/(self.num_of_imputation * self.lambda_))
            
            # set regression variables
            self.weight = []
            self.weight_a0 = []
            for i in range(self.num_of_imputation):
                self.weight_a0.append(pm3.Normal("alpha"+str(i), mu = 0, sigma = tt.sqrt(self.tg0)))
                self.weight.append(pm3.Normal("beta"+str(i), mu = 0, sigma = tt.sqrt(self.tg), shape = self.p) * self.pi)

            # set likelihood
            self.y_obs = []
            for j in range(self.num_of_imputation):
                self.y_obs.append(pm3.Normal("y_obs" + str(j), mu = self.weight_a0[j] + pm3.math.dot(self.X[j], self.weight[j]), sigma = tt.sqrt(self.sigma2), observed = self.y[j]))
  
    
  
if __name__ == "__main__":
    
    from bmiselect.utils.genDS_MAR import genDS_MAR
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
    gendata = genDS_MAR(n, p, 5, alpha_LM, miss_index,  covmat, beta, sigma, seed = 123) 
    X_array = gendata["M_X"]
    Y_array = gendata["M_Y"]
    model = Spike_laplace(Y_array, X_array)
    model.sample(10, 10, n_chain = 3)