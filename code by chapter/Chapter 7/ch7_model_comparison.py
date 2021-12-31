#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:26:59 2019

@author: tom verguts
AIC, BIC, and transfer-likelihood (i.e., cross-validation) calculation example
for the four-armed bandit problem (table 7.2)
"""

#%% import and initialize
import numpy as np
from ch6_generation import generate_learn, generate_learn2
from ch6_estimation import estimate_learn, estimate_learn2
from ch6_likelihood import logL_learn, logL_learn2

algorithm = "Powell"
learning_rate, learning_rate2, temperature = 0.8, 0.05, 1
ntrials = 1000
constant = 100000 # constant used in likelihood calculation


#%% actual simulation, estimation, and model comparison
for loop in range(2):
    if loop == 0:
        # data set 1: same learning rate for all trials
        data = generate_learn(alpha = learning_rate, beta = temperature, ntrials = ntrials, switch = True)
        data.insert(0, "index", range(ntrials))
        # for cross-validation purposes
        data_new = generate_learn(alpha = learning_rate, beta = temperature, ntrials = ntrials, switch = True)
        data_new.insert(0, "index", range(ntrials))
    else:    
        # data set 2: different learning rate for positive and negative trials
        data = generate_learn2(alpha1 = learning_rate, alpha2 = learning_rate2, beta = temperature, ntrials = ntrials, \
                               switch = True)
        data.insert(0, "index", range(ntrials))
        # for cross-validation purposes
        data_new = generate_learn2(alpha1 = learning_rate, alpha2 = learning_rate2, beta = temperature, ntrials = ntrials, \
                                   switch = True)
        data_new.insert(0, "index", range(ntrials))
        
    # model 1: same learning rate for all trials
    res = estimate_learn(nstim = 4, maxiter = 20000, data = data, algorithm = algorithm)
    Lik = res.fun*constant # note that ch6_likelihood.py actually calculates minus log-likelihood...
    est_par = res.x
    print(est_par)
    AIC = 2*Lik + 2*est_par.size
    BIC = 2*Lik + np.log(ntrials)*est_par.size
    cross_val = logL_learn(parameter = est_par, data = data_new)*constant
    print("Model 1: -log L = {0:.3f}, AIC = {1:.2f}, BIC = {2:.2f}, cross-val = {3:.2f}".format(Lik, AIC, BIC, cross_val))

    # model 2: different learning rate for positive and negative rewards
    res2 = estimate_learn2(nstim = 4, maxiter = 20000, data = data, algorithm = algorithm)
    Lik2 = res2.fun*constant
    est_par2 = res2.x
    print(est_par2)
    AIC = 2*Lik2 + 2*est_par2.size
    BIC = 2*Lik2 + np.log(ntrials)*est_par2.size
    cross_val = logL_learn2(parameter = est_par2, data = data_new)*constant
    print("Model 2: -log L = {0:.3f}, AIC = {1:.2f}, BIC = {2:.2f}, cross-val = {3:.2f}".format(Lik2, AIC, BIC, cross_val))