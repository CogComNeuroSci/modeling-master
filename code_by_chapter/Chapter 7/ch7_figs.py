#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:19:06 2019

@author: tom verguts
pics and tables for chapter 7, testing computational models
table 7.2 is generated with file ch7_model_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt

#%% fig 7.2; illustrating statistical dependency
dep = [0, 0.8, 1] # trial-by-trial dependencies
n_trials = 100
titles = ["Fig 7.2a: No dependency", "Fig 7.2b: Dependency", "Fig 7.2c: Strong dependency"]

fig, axs = plt.subplots(nrows = 1, ncols = 3)
x = range(n_trials)
y = np.empty(n_trials)

for plot_loop in range(3):
    for step in range(n_trials-1):
        y[step+1] = dep[plot_loop]*y[step] + np.random.randn()*0.01
    y_hat = y - np.mean(y)    # the residuals
    axs[plot_loop].scatter(x, y_hat, color = "black")
    axs[plot_loop].set_title(titles[plot_loop])
    axs[plot_loop].set_xlabel("Trial nr")
	
	
#%% 
"""
table 7.1: illustration of AIC, BIC, and transfer test
AIC, BIC, and transfer loglikelihood illustration for coin tossing model
model 1: There is just one p-value
model 2: Part 1 and part 2 have a different p-value
"""
n_trials = 1000
p = [[0.6, 0.6], [0.5, 0.7]]
K = [1, 2] # nr of parameters for model 1 (K[0]) and for model 2 (K[1])

for loop in range(2): # ranges over p-values (loop = 0: p equal for part 1 and 2; loop = 1: p different for part 1 and 2)
    data = np.zeros(n_trials)
    data_new = np.zeros(n_trials) # these are the transfer data
    for data_loop in range(n_trials):
        data[data_loop] = (np.random.rand() > p[loop][data_loop//(n_trials//2)])
        data_new[data_loop] = (np.random.rand() > p[loop][data_loop//(n_trials//2)]) # new data set for the transfer test
    n_heads = np.array([np.sum(data[:n_trials//2+1]), np.sum(data[n_trials//2:])])
    n_heads_new = np.array([np.sum(data_new[:n_trials//2+1]), np.sum(data_new[n_trials//2:])])
    p1 = np.sum(n_heads)/data.size           # p-value estimates only on data (not on transfer data data_new)
    p2 = np.array(n_heads/(data.size//2))    # p-value estimates according to model 2: part 1 and part 2 have different estimates

    lik1 = np.sum(n_heads)*np.log(p1) + (n_trials-np.sum(n_heads))*np.log(1-p1)     # likelihood of model 1
    lik2 = ( n_heads[0]*np.log(p2[0]) + (n_trials//2-n_heads[0])*np.log(1-p2[0])+   # likelihood of model 2
            n_heads[1]*np.log(p2[1]) + (n_trials//2-n_heads[1])*np.log(1-p2[1]) )

    aic1 = -2*lik1 + 2*K[0] # AIC for model 1
    aic2 = -2*lik2 + 2*K[1] # AIC for model 2

    bic1 = -2*lik1 + K[0]*np.log(n_trials) # BIC for model 1
    bic2 = -2*lik2 + K[1]*np.log(n_trials) # BIC for model 2

    lik1_new = np.sum(n_heads_new)*np.log(p1) + (n_trials-np.sum(n_heads_new))*np.log(1-p1)     # transfer likelihood for model 1
    lik2_new = ( n_heads_new[0]*np.log(p2[0]) + (n_trials//2-n_heads_new[0])*np.log(1-p2[0])+   # transfer likelihood for model 2
            n_heads_new[1]*np.log(p2[1]) + (n_trials//2-n_heads_new[1])*np.log(1-p2[1]) )
    print("Dataset {0}\n LogL model1: {1:.2f} model 2: {2:.2f}\n AIC model1: {3:.2f} \
           model2: {4:.2f}\n BIC model1 {5:.2f} BIC model2 {6:.2f} LogLtransf model1: {7:.2f} LogLtransf model2: {8:.2f}"
		   .format(loop+1,lik1, lik2, aic1, aic2, bic1, bic2, lik1_new, lik2_new))
		   
