#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:32:55 2018

@author: tom verguts
defines the log-likelihoods of the logit and learning models
"""

import pandas as pd
import numpy as np


def logit(beta_in,x1, x2):
    return 1/(1+np.exp(beta_in*(x2-x1)))

# likelihood for the alpha-beta model
def logL_ab(parameter, nstim, file_name): 
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

# likelihood for the learning model
def logL_learn(parameter, nstim, file_name): 
    data = pd.read_csv(file_name)
    ntrials = data.shape[0]
    # calculate log-likelihood
    logLik = 0
    value = np.random.rand(nstim)
    for trial_loop in range(ntrials):
        logLik = logLik + (
                np.log( logit(parameter[1],value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]],
                                        value[data.iloc[trial_loop,1-data.iloc[trial_loop,3]+1]]) ) )
        value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] =(
                value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]] +
                parameter[0]*(data.iloc[trial_loop,4]-value[data.iloc[trial_loop,data.iloc[trial_loop,3]+1]]) )
    return -logLik