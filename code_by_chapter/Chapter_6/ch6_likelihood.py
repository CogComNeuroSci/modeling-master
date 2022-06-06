#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:32:55 2018

@author: tom verguts
defines the log-likelihoods of the alpha-beta (ie, loglinear) and learning models

TBD:
the alpha-beta formulation can be improved by coding alpha and beta as parameters from min to plus infinity
(instead of in range (0,1), as is done now, which leads to instability
 
the learning model formulation logL_learn was used in the MCP book reported simulations;
the formulation logL_learnR is an improved formulation: It's more Robust because it avoids exponentiation as much as possible
"""

import pandas as pd
import numpy as np


def logit(beta_in,x1, x2):
    return 1/(1+np.exp(beta_in*(x2-x1)))


def logL_ab(parameter, nstim, file_name): 
    """ likelihood for the alpha-beta model"""
    data = pd.read_csv(file_name)
    ntrials = data.shape[0]
    # calculate log-likelihood
    logLik = 0
    for trial_loop in range(ntrials):
        logLik = logLik + (
                data.iloc[trial_loop,2]*np.log(parameter[0]) +
                data.iloc[trial_loop,1]*data.iloc[trial_loop,2]*np.log(parameter[1]) +
                (1-data.iloc[trial_loop,2])*np.log(1-parameter[0]*(parameter[1]**data.iloc[trial_loop,1]))
                )
    return -logLik    

  
def logL_learn(parameter = [0.6, 1], nstim = 5, file_name = "", data = None, prior = (0, 0), startvalue = 0): 
    """likelihood for the learning model
	parameter = learning rate, temperature
    prior = (mean, precision); higher precision (> 0) gives more weight to the prior"""  
    if len(file_name)>0:
        data = pd.read_csv(file_name)
    else:
        data = data
    ntrials = data.shape[0]
    # calculate log-likelihood
    logLik = 0
    value = np.random.rand(nstim)
    for trial_loop in range(ntrials):
        logLik = logLik + (
                np.log( logit(parameter[1],value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]],
                                        value[data.iloc[trial_loop,1-data.iloc[trial_loop,3]+1]]) ) )
        prediction_error = parameter[0]*(data.iloc[trial_loop,4]-value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]])
        value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] = (
                value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] + prediction_error)
    logLik = logLik - (prior[1]/np.sqrt(2*np.pi))*( (parameter[0]-prior[0])**2 + (parameter[1]-prior[0])**2 )  
    return -logLik/100000

def logL_learnR(parameter = [0.6, 1], nstim = 5, file_name = "", data = None, prior = (0, 0), startvalue = 0): 
    """Robust version of the likelihood for the learning model
    this code avoid the exponentiation as much as possible
	parameter = learning rate, temperature
    prior = (mean, precision); higher precision (> 0) gives more weight to the prior"""  
    if len(file_name)>0:
        data = pd.read_csv(file_name)
    else:
        data = data
    ntrials = data.shape[0]
    # calculate log-likelihood
    logLik = 0
    value = np.random.rand(nstim)
    for trial_loop in range(ntrials):
        v_chosen =   value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]]
        v_unchosen = value[data.iloc[trial_loop,2-data.iloc[trial_loop,3]]]						 
        max_v = np.maximum(v_chosen, v_unchosen)
        logLik = logLik + parameter[1]*v_chosen - parameter[1]*max_v - np.log(np.exp(parameter[1]*(v_chosen-max_v)) + np.exp(parameter[1]*(v_unchosen-max_v)))
        prediction_error = parameter[0]*(data.iloc[trial_loop,4]-value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]])
        value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] = (
                value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] + prediction_error)
    logLik = logLik - (prior[1]/np.sqrt(2*np.pi))*( (parameter[0]-prior[0])**2 + (parameter[1]-prior[0])**2 )  
    return -logLik

def logL_learn2(parameter = [0.6, 0.3, 1], nstim = 5, file_name = "", data = None, prior = (0, 0), startvalue = 0): 
    """likelihood for the learning model with two learning rates
	parameter = learning rate 1, learning rate 2, temperature
    prior = (mean, precision); higher precision (> 0) gives more weight to the prior"""  
    if len(file_name)>0:
        data = pd.read_csv(file_name)
    else:
        data = data
    ntrials = data.shape[0]
    # calculate log-likelihood
    logLik = 0
    value = np.random.rand(nstim)
    for trial_loop in range(ntrials):
        if data.iloc[trial_loop,4]==1:
            learning_rate = parameter[0]
        else:
            learning_rate = parameter[1]
        prediction_error = learning_rate * (data.iloc[trial_loop,4]-value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]])
        logLik = logLik + (
                np.log( logit(parameter[2],value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]],
                                        value[data.iloc[trial_loop,1-data.iloc[trial_loop,3]+1]]) ) )
        value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] = (
                value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] + prediction_error)
    logLik = logLik - \
            (prior[1]/np.sqrt(2*np.pi))*( (parameter[0]-prior[0])**2 + (parameter[1]-prior[0])**2 + (parameter[2]-prior[0])**2 )  
    return -logLik/100000