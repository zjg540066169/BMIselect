#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:58:59 2021

This function is used to generate data under Miss At Random(MAR) assumption.

@author: Jungang Zou
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation import mice
import time


def genDS_MAR(n, p, n_imp, alpha, miss_index, covmat, beta, sigma, cat = False, seed = None):
    # Input parameters:
    # n: number of samples.
    # p: number of covariates.
    # n_imp: number of generated multiply-imputation set.
    # alpha: a matrix of coefficients to calculate a logistic function to generate missingness. More details can be found in our manuscript.
    # missing index: a numerical vector to indicate which variables are missing. For example: [2,5,10] indicates 2ed, 5th, 10th variables are missing.
    # covmat: covariance matrix to generate covariates.
    # beta: regression coefficients to generate response variable.
    # sigma: regression variance to generate response variable.
    # cat: indicate if generated covariates are binary. If cat is True, then X = 0 if X < 0; and X = 1 if X >= 0.
    # seed: random seed.
    
    # Output parameters:
    # a dictionary:
        # "M"  : multiply imputed data with a long data frame, dim = (n * n_imp, p + 1)
        # "M_X": multiply imputed covariates, dim = (n_imp, n, p)
        # "M_Y": multiply imputed response variable, dim = (n_imp, n)
        # “C”  : complete case only, the missing samples are deleted.
        # "O"  : original data without missing values
        # "avg": averaged each covariate across multiply-imputed data
        # "ncom" : number of complete cases
    if seed is None:
        seed = time.time()
    np.random.seed(seed)
    
    # compound symmetric correlation matrix with off-diagonal value of "corr"
    C = np.random.multivariate_normal(mean = np.repeat(0, p), cov = covmat, size = n)

    if cat:
        C = C >= 0
        C = C.astype(np.float)
    # generate values of Y
    Y = C.dot(beta) + sigma * np.random.normal(size = n)

    # drop values of variables in incompl with probability dtermined by obs
    
    obs = C[:, np.setdiff1d(np.arange(p), miss_index)].copy()
    incompl = C[:, miss_index].copy()
  

  
    for j in range(int(p/2)): 
        U = np.random.uniform(size = n)
        miss_fr = 1/     (     1+np.exp(    - alpha[j,:].dot( np.transpose(np.concatenate((np.repeat(1,n).reshape(-1, 1),obs,Y.reshape(-1, 1)), axis=1) ))   )    )
        incompl[U <= miss_fr,j] = np.nan
    
    I = np.concatenate((obs, incompl), axis = 1)
    
  
    ds_I = pd.DataFrame(np.concatenate((Y.reshape(-1, 1), I), axis = 1), columns= ['y'] + ["x"+str(i) for i in range(p)])
    
    
    imp = mice.MICEData(ds_I)
    fml = ''
    for i in range(p - 1):
        fml += "x" + str(i) +" + "
    fml += "x" + str(p - 1)
    imputed_dataset = []
    if cat:
        imp.set_imputer('y', formula=fml, model_class = sm.Logit)
    else:
        imp.set_imputer('y', formula=fml)
    for j in range(n_imp):
        imp.update_all(n_imp)
        data = imp.data.copy()
        data.loc[:, "id"] = data.index
        data.loc[:, "imp"] = j
        imputed_dataset.append(data)
    
    
    
    # multiple imputation
    M = pd.concat(imputed_dataset, axis = 0)
    
    
    # take an average of imputed values for missing data
    avg = M.iloc[:, 0:(p+1)].mean(0)
    

    # Calculate the number of complete cases
    complete = np.logical_not(np.isnan(I.sum(1)))
    n_C = complete.sum()
    
    # Output data sets
    ds_M = M  # multiply imputed data with a long data frame
    X_array = []
    Y_array = []
    for i in range(n_imp):
        X_array.append(ds_M[ds_M.loc[:, "imp"] == i].iloc[:, 1:(p + 1)].to_numpy())
        Y_array.append(ds_M[ds_M.loc[:, "imp"] == i].iloc[:, 0].to_numpy())
    X_array = np.array(X_array)
    Y_array = np.array(Y_array)
    ds_C = ds_I[complete]  # data with complete cases only
    ds_O = np.concatenate((Y.reshape(-1, 1), C), axis = 1)  # original data without missing values
    ds_avg = avg # data with average imputed data
    
  
    return {"M" : ds_M, "M_X": X_array, "M_Y": Y_array, "C" : ds_C, "O" : ds_O, "avg" : ds_avg, "ncom" : n_C}


if __name__ == "__main__":
    s = 123
    n = 100 ## sample size
    p = 40 ## of variables
    K = 5  ## number of imputations
    W = np.diag(np.repeat(1, n)) ## identity weight matrix
    rho = 0.1
    SNR = 3
    important = np.array([1,2,3,11,12,13]) - 1
    noise = np.concatenate([np.arange(3, 10), np.arange(13, 20)])
    
    
    
    miss_index = np.append(np.arange(10, 20), np.arange(30, 40))
    keep_index = np.setdiff1d(np.arange(p), miss_index)
    important = np.array([1,2,5,11,12,15, 21,22,25, 31,32,35]) - 1
    noise = np.setdiff1d(np.arange(p), important)
    truth = np.array([False for i in range(p)])
    truth[important] = True
    beta = np.zeros(p)
    beta[important] = 1
    
    # compound symmetric correlation matrix with off-diagonal value of "corr"
    covmat = np.repeat(rho, p * p).reshape(p, p)
    di = np.diag_indices(p)
    covmat[di] = 1
    
   
    sigma = np.sqrt(beta.dot(covmat).dot(beta/SNR))
    
    alpha_LM = np.array([
            -4,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0.5,
            -4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0.5
            ]).reshape(20, 22)
    
    data1 = genDS_MAR(n, p, 5, alpha_LM, miss_index, covmat, beta, sigma, seed = s)

