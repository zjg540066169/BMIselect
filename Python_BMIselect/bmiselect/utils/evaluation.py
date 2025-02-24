#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:35:23 2022

This file provides a set of evaluation functions, including:
    * Sensitivity / Recall
    * Specificity
    * Precision
    * F1-score
    * MSE
    
Also some functions using selected variables to refit linear regression models are also provided:
    * Refitting LR
    * Pooled coefficients
    * Pooled covariance matrix


@author: Jungang Zou
"""
import numpy as np
import statsmodels.api as sm


def sensitivity(truth, select):
    # Calculate the sensitivity given truly important variables and selected variables. 
    # The input should be binary array.
    return np.sum(np.bitwise_and(truth, select)) / np.sum(truth)


def specificity(truth, select):
    # Calculate the specificity given truly important variables and selected variables. 
    # The input should be binary array.
    return np.sum(np.bitwise_and(1 - truth, 1 - select)) / np.sum(1 - truth)


def distance(truth, select):
    # Calculate the distance given truly important variables and selected variables. 
    # The input should be binary array.
    # Distance = sqrt(sensitivity^2 + specificity^2)
    return np.sqrt(np.power(sensitivity(truth, select), 2) + np.power(specificity(truth, select), 2))


def precision(truth, select):
    # Calculate the precision given truly important variables and selected variables. 
    # The input should be binary array.
    return np.sum(np.bitwise_and(truth, select)) / np.sum(select)


def recall(truth, select):
    # Calculate the recall given truly important variables and selected variables. 
    # The input should be binary array.
    return np.sum(np.bitwise_and(truth, select)) / np.sum(truth)


def f1_score(truth, select):
    # Calculate the f1-score given truly important variables and selected variables. 
    # The input should be binary array.
    pre = precision(truth, select)
    sen = sensitivity(truth, select)
    return 2 * (pre * sen) / (pre + sen)
  
    
def fit_lr(select, X, Y, intercept = True):
    # Refit linear regressions by using selected variables.
    # There is a list of statsmodels.regression.linear_model.RegressionResults objects returned.
    # The length of output is equal to the number of imputation.
    # Parameter "intercept" controls whether to include intercept in each refitted regression.
    params = []
    X_select = X[:, :, select]
    for i in range(X_select.shape[0]):
        if intercept:
            X_i = sm.add_constant(X_select[i, :, :])
        else:
            X_i = X_select[i, :, :]
        Y_i = Y[i, :]
        model = sm.OLS(Y_i, X_i)
        results = model.fit()
        params.append(results)
    return params


def pooled_coefficients(select, X, Y, intercept = True):
    # Refit linear regressions by using selected variables.
    # Return the pooled coefficients by using Rubin`s Rule.
    # Parameter "intercept" controls whether to include intercept in each refitted regression.
    coefficients_select = []
    X_select = X[:, :, select]
    for i in range(X_select.shape[0]):
        if intercept:
            X_i = sm.add_constant(X_select[i, :, :])
        else:
            X_i = X_select[i, :, :]
        Y_i = Y[i, :]
        model = sm.OLS(Y_i, X_i)
        results = model.fit()
        coefficients_select.append(results.params)
    #pool_coeff = np.zeros(select.shape)
   # pool_coeff[select] = np.mean(coefficients_select, 0)[1:]
    #return np.append(np.mean(coefficients_select, 0)[0], pool_coeff)
    return np.mean(coefficients_select, 0)


def pooled_covariance(select, X, Y, intercept = True):
    # Refit linear regressions by using selected variables.
    # Return the pooled covariance matrix by using Rubin`s Rule.
    # Parameter "intercept" controls whether to include intercept in each refitted regression.
    covariance_select = []
    bet_covariance = []
    X_select = X[:, :, select]
    for i in range(X_select.shape[0]):
        if intercept:
            X_i = sm.add_constant(X_select[i, :, :])
        else:
            X_i = X_select[i, :, :]
        Y_i = Y[i, :]
        model = sm.OLS(Y_i, X_i)
        results = model.fit()
        covariance_select.append(results.cov_params())
        bet_covariance.append(results.params.reshape(-1, 1).dot(results.params.reshape(1, -1)))
    with_in = np.array(covariance_select).mean(0)
    between = np.sum(bet_covariance, 0) / (X.shape[0] - 1)
    return with_in + (1 + 1/X.shape[0]) * between


def mse(beta, covariance, select, X, Y, intercept = True):
    # Refit linear regressions by using selected variables.
    # Return MSE by using Rubin`s Rule and the covariance matrix of ground truth.
    # Parameter "intercept" controls whether to include intercept in each refitted regression.
    if np.sum(select) == 0:
        pool_beta = np.zeros(beta.shape)
    else:
        coefficients_select = []
        X_select = X[:, :, select]
        for i in range(X_select.shape[0]):
            if intercept:
                X_i = sm.add_constant(X_select[i, :, :])
            else:
                X_i = X_select[i, :, :]
            Y_i = Y[i, :]
            model = sm.OLS(Y_i, X_i)
            results = model.fit()
            coefficients_select.append(results.params)
        pool_beta = np.zeros(select.shape)
        if intercept:
            pool_beta[select] = np.mean(coefficients_select, 0)[1:]
        else:
            pool_beta[select] = np.mean(coefficients_select, 0)
            
    return (pool_beta - beta).dot(covariance).dot(pool_beta - beta)

# def mse(beta, select, X, Y):
#     if np.sum(select) == 0:
#         true_beta = 
#     else:
#         pool_beta = pooled_coefficients(select, X, Y)
#         if pool_beta.shape != beta[select].shape:
#             true_beta = np.append(0, beta[select])
#         elif pool_beta.shape == beta[select].shape:
#             true_beta = beta[select]
        
#     pool_cov = pooled_covariance(select, X, Y)
#     return (pool_beta - true_beta).dot(pool_cov).dot(pool_beta - true_beta)
