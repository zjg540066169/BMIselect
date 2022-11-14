#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 08:31:22 2022

@author: Jungang Zou
"""

if __name__ == "__main__":
    
    from bmiselect.models.ARD import ARD
    from bmiselect.models.Horseshoe import Horseshoe
    from bmiselect.models.Spike_ridge import Spike_ridge
    from bmiselect.models.Ridge import Ridge
    from bmiselect.models.Laplace import Laplace
    from bmiselect.models.Spike_laplace import Spike_laplace

    from bmiselect.utils.evaluation import *
    from bmiselect.utils.genDS_MAR import genDS_MAR
    from bmiselect.utils.genDS_MCAR import genDS_MCAR


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    
    
    
    
    # parameters for generating data
    n = 100
    p = 20
    K = 5
    W = np.diag(np.repeat(1, n))
    rho = 0.1
    SNR = 3
    important = np.array([1,2,3,11,12,13]) - 1
    noise = np.concatenate([np.arange(3, 10),np.arange(13, 20) ])
    
    # ground truth for important variables
    truth = np.array([False for i in range(20)])
    truth[important] = True
    
    # compound symmetric correlation matrix with off-diagonal value of "corr"
    covmat = np.ones((p,p)) * rho
    covmat[np.diag_indices_from(covmat)] = 1
    
    
    # coefficients and variance
    beta = np.array([0.5,1,2,0,0,0,0,0,0,0,0.5,1,2, 0,0,0,0,0,0,0])
    sigma = np.sqrt(beta.dot(covmat).dot(beta)/SNR) 
    true_index = np.array([1, 1, 1, 0,0,0,0,0,0,0,1, 1, 1, 0,0,0,0,0,0,0])
    
    # low missing proportion
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
    
    # high missing proportion
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
    
    # generate data with MAR assumption
    miss_index = np.arange(10, 20)
    gendata = genDS_MAR(n, p, 5, alpha_LM, miss_index,  covmat, beta, sigma, seed = 123) 
    
    # get X and Y
    X_array = gendata["M_X"]
    Y_array = gendata["M_Y"]
    
    # build model and sample posterior samples.
    
    # shrinkage model
    model1 = Laplace(Y_array, X_array, standardize = True, r = 2, s = 15)
    # model1 = Horseshoe(Y_array, X_array, standardize = True)
    # model1 = ARD(Y_array, X_array, standardize = True)
    model1.sample(n_post = 10, n_burn = 5, target_accept = 0.9, n_chain = 3, n_thread = 4, max_treedepth = 10, seed = 123)
    summary_stats1 = model1.summary()
    print(summary_stats1)
    select1 = model1.select(0.95)
    # get posterior samples and visualization
    alpha_posterior = model1.get_posterior_samples("alpha").reshape(-1)
    sns.kdeplot(alpha_posterior, shade=True)
    plt.plot()      
            


    
    
    # discrete-mixture model
    model2 = Spike_laplace(Y_array, X_array, standardize = True, lambda_ = 6/11, a = 1, b = 1)
    # model2 = Spike_ridge(Y_array, X_array, standardize = True, p0 = 0.5)
    model2.sample(n_post = 10, n_burn = 5, target_accept = 0.9, n_chain = 3, n_thread = 4, max_treedepth = 10, seed = 123)
    summary_stats2 = model2.summary()
    print(summary_stats2)
    select2 = model2.select(0.5)
    
    # get posterior samples and visualization
    beta_posterior = model2.get_posterior_samples("beta").reshape(-1, p)
    beta_posterior_long = pd.melt(pd.DataFrame(beta_posterior, columns=np.arange(p)))
    # keep important variables
    beta_posterior_long = beta_posterior_long[beta_posterior_long["variable"].isin(np.where(select2)[0])]
    g = sns.FacetGrid(beta_posterior_long, row="variable", aspect=np.sum(select2), height=1.2)
    g.map_dataframe(sns.kdeplot, x="value")
    plt.plot()
    
    
    # evaluation
    print("sensitivity of Multi-Laplace:", sensitivity(select = select1, truth = truth))
    print("MSE of Multi-Laplace:", mse(beta, covariance = covmat, select = select1, X = X_array, Y = Y_array, intercept = True))
    
    print("specificity of Spike-Laplace:", specificity(select = select2, truth = truth))
    print("f1_score of Spike-Laplace:", f1_score(select = select2, truth = truth))
    
    
    # refit linear regression by using selected variabels
    lr_models = fit_lr(select1, X_array, Y_array)
    for lr in lr_models:
        print(lr.summary())
        
    # get pooled coefficients estimates by using Rubin`s rule
    lr_coef = pooled_coefficients(select2, X_array, Y_array)
    print(lr_coef)
    
    
    # get pooled covariance estimates by using Rubin`s rule
    lr_covariance = pooled_covariance(select2, X_array, Y_array)
    print(lr_covariance)