#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:06:17 2018

@author: tom verguts
plots for the exercises for the exam
x.y is for chapter x, exercise y
"""
import numpy as np
import matplotlib.pyplot as plt

#%% exercise 1.1
fig, axs = plt.subplots(nrows = 2, ncols = 2)
beta = [0.2, 1, 1.5, 0.01]
n_trials = 40

X = 3 # value to be estimated
for beta_loop in range(len(beta)):
    X_est = np.zeros(n_trials+1)
    for trial_loop in range(n_trials):
        X_est[trial_loop+1] = X_est[trial_loop] + beta[beta_loop]*(X - X_est[trial_loop])
    axs[beta_loop%2, int(np.floor(beta_loop/2))].plot(range(n_trials+1), X_est)