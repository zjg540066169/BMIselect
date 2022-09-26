#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 23:04:28 2021

@author: jungang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.imputation import mice






def genDS_MCAR(s, miss_index, n, p, covmat, beta, sigma, miss_fr = 0.05, cat = False):
    np.random.seed(s)
    
    # compound symmetric correlation matrix with off-diagonal value of "corr"
    C = np.random.multivariate_normal(mean = np.repeat(0, p), cov = covmat, size = n)

    if cat:
        C = C >= 0
        C = C.astype(np.float)

    # generate values of Y
    Y = C.dot(beta) + sigma * np.random.normal(size = n)

    # drop values of variables in incompl with probability dtermined by obs
    

    I = C.copy()
  
    for j in miss_index: 
        U = np.random.uniform(size = n)
        #miss_fr = 1/     (     1+np.exp(    - alpha[j,:].dot( np.transpose(np.concatenate((np.repeat(1,n).reshape(-1, 1),obs,Y.reshape(-1, 1)), axis=1) ))   )    )
        I[U <= miss_fr,j] = np.nan
    
   
    
  
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
    for j in range(5):
        imp.update_all(5)
        data = imp.data.copy()
        data.loc[:, "id"] = data.index
        data.loc[:, "imp"] = j
        imputed_dataset.append(data)
    
    
    
    # multiple imputation
    #IMP = mice(ds.I, meth="norm", seed=s, printFlag=FALSE)
    #M = complete(IMP, "long")
    #M = cbind(M[,3:23],M[,1:2])
    M = pd.concat(imputed_dataset, axis = 0)
    
    
    # take an average of imputed values for missing data
    avg = M.iloc[:, 0:(p+1)].mean(0)
    

    # Calculate the number of complete cases
    complete = np.logical_not(np.isnan(I.sum(1)))
    n_C = complete.sum()
    
  
    
  
    # Output data sets
    ds_M = M  # multiply imputed data with a long data frame
    ds_C = ds_I[complete]  # data with complete cases only
    ds_O = np.concatenate((Y.reshape(-1, 1), C), axis = 1)  # original data without missing values
    ds_avg = avg # data with average imputed data
    
  
    return {"M" : ds_M, "C" : ds_C, "O" : ds_O, "avg" : ds_avg, "ncom" : n_C}


if __name__ == "__main__":
    s = 123
    n = 100 ## sample size
    p = 40 ## of variables
    K = 5  ## number of imputations
    W = np.diag(np.repeat(1, n)) ## identity weight matrix
    rho = 0.1
    SNR = 3

    
    
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
    
   
    data1 = genDS_MCAR(s,  miss_index, n, p, covmat, beta, sigma, miss_fr = 0.05, cat = True)
