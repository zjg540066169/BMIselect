#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:38:18 2022

This is the main class of all Bayesian MI-LASSO models.
Some important functions are provided:
    * sample
    * summary
    * select
    * get_posterior_samples

@author: Jungang Zou
"""

import pymc3 as pm3
import theano.tensor as tt
import numpy as np
from theano import printing, function
import arviz as az
import time
import xarray as xr

class Bmi_lasso:
    def __init__(self, y, X, standardize = True):
        # initialization
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
        self.type_model = None
        self.chains = None
            
    def summary(self, rescale = True):
        # statistical summary for MCMC samples
        if self.trace is None:
            raise Exception("Please use function model.sample to run the MCMC chains before calling this function!")
        if rescale:
            return(az.summary(self.rescaled_trace))
        else:
            return(az.summary(self.trace))
        
    def __rescale_coefficients__(self):
        # if coefficients are standardized, this function is used to rescale standardized coefficients into the original scale.
        if self.rescaled is None:
            pass
        if self.rescaled == True:
            pass
        for i in range(self.num_of_imputation):
            self.rescaled_trace.posterior["beta" + str(i)].values = self.rescaled_trace.posterior["beta" + str(i)].values / self.x_std[i,:].reshape(1, 1, -1)
        self.rescaled == True
        
        
    def sample(self, n_post, n_burn, target_accept = 0.9, n_chain = 1, n_thread = 4, max_treedepth = 10, seed = None):
        
        if self.standardize:
            self.rescaled = False
        else:
            self.rescaled = None
        
        # sample MCMC chains.
        if seed is None:
            seed = int(time.time())
        with self.model:
            try:
                self.trace = pm3.sample(draws = n_post, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = n_burn, target_accept = target_accept, max_treedepth = 10, return_inferencedata=True)
            except ValueError:
                self.trace = pm3.sample(draws = n_post, random_seed = seed, cores = n_thread, progressbar = False, chains = n_chain, tune = n_burn, return_inferencedata=True)
        self.chains = n_chain
      #  if self.chains > 1:
      #      self.trace.posterior = az.extract(self.trace.posterior)
#
        
        # rescale   
        if self.rescaled is None:
            self.rescaled_trace = self.trace
        if self.rescaled == False:
            self.rescaled_trace = self.trace.copy()
            self.__rescale_coefficients__()

        # Bayesian mixture
        #if self.type_model == "shrinkage":
        beta = []
        alpha = []
        rescaled_beta = []
        rescaled_alpha = []
        for i in range(self.num_of_imputation):
           beta.append(self.trace.posterior["beta" + str(i)].values)
           alpha.append(self.trace.posterior["alpha" + str(i)].values)
           rescaled_beta.append(self.rescaled_trace.posterior["beta" + str(i)].values)
           rescaled_alpha.append(self.rescaled_trace.posterior["alpha" + str(i)].values)
        
        alpha = np.concatenate(alpha, 1)
        beta = np.concatenate(beta, 1)
        rescaled_alpha = np.concatenate(rescaled_alpha, 1)
        rescaled_beta = np.concatenate(rescaled_beta, 1)
        
        chain_var = self.trace.posterior["beta0"].coords["chain"].copy()

        
        self.trace.posterior["beta"] = xr.DataArray(data = beta, dims = ("chain", "draw", "beta_dim_0" ), coords = {"beta_dim_0":np.arange(self.p), "chain":chain_var, "draw": np.arange(beta.shape[1])})
        self.trace.posterior["alpha"] = xr.DataArray(data = alpha, dims = ("chain", "draw"), coords = {"chain":chain_var, "draw": np.arange(alpha.shape[1])})
        
        self.rescaled_trace.posterior["beta"] = xr.DataArray(data = rescaled_beta, dims = ("chain", "draw", "beta_dim_0" ), coords = {"beta_dim_0":np.arange(self.p), "chain":chain_var, "draw": np.arange(beta.shape[1])})
        self.rescaled_trace.posterior["alpha"] = xr.DataArray(data = rescaled_alpha, dims = ("chain", "draw"), coords = {"chain":chain_var, "draw": np.arange(rescaled_alpha.shape[1])})
        
        
            
    def select(self, value, rescale = True):
        # value:
        #  the CI selection criteria for shrinkage models.
        #  the cutting-point for discrete models.
        if value > 1 or value < 0:
            raise Exception("Please make the value between [0, 1]")
        self.value = value
        if self.type_model == "shrinkage":
            if rescale == False:
                lower_quantile = np.quantile(az.extract(self.trace.posterior)["beta"].values, (1 - self.value) / 2, 1)
                upper_quantile = np.quantile(az.extract(self.trace.posterior)["beta"].values, 1 - (1 - self.value) / 2, 1)
            else:
                lower_quantile = np.quantile(az.extract(self.rescaled_trace.posterior)["beta"].values, (1 - self.value) / 2, 1)
                upper_quantile = np.quantile(az.extract(self.rescaled_trace.posterior)["beta"].values, 1 - (1 - self.value) / 2, 1)
            
            select = np.bitwise_not(np.bitwise_and(lower_quantile <= 0, upper_quantile >= 0))
        else:
            if rescale == False:
                select = (np.median(az.extract(self.trace.posterior)["g"].values, 1) > self.value)
            else:
                select = (np.median(az.extract(self.rescaled_trace.posterior)["g"].values, 1) > self.value)
        return select
        
        
    def get_posterior_samples(self, var_name, rescale = True):
        # return posterior samples, dim = (chain, n_post, p)
        if rescale:
            return self.rescaled_trace.posterior[var_name].values
        else:
            return self.trace.posterior[var_name].values
        
        
        
        
            
            