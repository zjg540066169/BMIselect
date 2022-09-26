#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:38:18 2022

@author: zoujungang
"""

import pymc3 as pm3
import theano.tensor as tt
import numpy as np
from theano import printing, function
import arviz as az
import time

class Bmi_lasso:
    def __init__(self, y, X, standardize = True):
        self.y = y
        self.X = X
        self.standardize = standardize
        if type(self.X) == list:
            self.X = np.array(self.X)
        if type(self.y) == list:
            self.y = np.array(self.y)            
        self.num_of_imputation = self.X.shape[0]
        self.n = self.X.shape[1]
        self.p = self.X.shape[2]
        self.rescaled = None
        if self.standardize:
            self.rescaled = False
            self.y_mean = self.y.mean(1)
            self.x_mean = []
            self.x_std = []
            for i in range(self.num_of_imputation):
                self.x_mean.append(self.X[i,:,:].mean(0))
                self.x_std.append(self.X[i,:,:].std(0))
                self.X[i,:,:] = (self.X[i,:,:] - self.x_mean[i]) / self.x_std[i]
            self.x_mean = np.array(self.x_mean)
            self.x_std = np.array(self.x_std)
        self.model = pm3.Model()
        self.trace = None
        self.rescaled_trace = None
        self.weight = []
            
    def summary(self, origina_scale = True):
        if self.trace is None:
            raise Exception("Please use function model.sample to run the MCMC chains before calling this function!")
        if origina_scale:
            return(az.summary(self.rescaled_trace))
        else:
            return(az.summary(self.trace))
        
    def rescale_coefficients(self):
        if self.rescaled is None:
            pass
        if self.rescaled == True:
            pass
        for i in range(self.num_of_imputation):
            self.rescaled_trace.posterior["beta" + str(i)].values = self.rescaled_trace.posterior["beta" + str(i)].values / self.x_std[i,:].reshape(1, 1, -1)
        self.rescaled == True
        
        
    def sample(self, n_post, n_burn, target_accept = 0.9, n_chain = 1, n_thread = 4, seed = None):
        if seed is None:
            seed = int(time.time())
        with self.model:
            try:
                self.trace = pm3.sample(draws = n_post, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = n_burn, target_accept = target_accept, max_treedepth = 10, return_inferencedata=True)
            except ValueError:
                self.trace = pm3.sample(draws = n_post, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = n_burn, return_inferencedata=True)
        if self.rescaled == False:
            self.rescaled_trace = self.trace.copy()
            self.rescale_coefficients()
        if self.rescaled is None:
            self.rescaled_trace = self.trace