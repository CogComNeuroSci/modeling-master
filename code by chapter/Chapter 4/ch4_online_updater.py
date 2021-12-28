#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 18:11:31 2021

@author: tom verguts
simple online updater of a rescorla-wagner type rule
can be used to check convergence properties for different update (learning) rates
"""

import numpy as np
import matplotlib.pyplot as plt

n_trials = 200
gamma = 1
mean, std = 1, 0.00
beta_list = [0.1, 1, 1.2, -0.5] # list of update (learning) rates to be checked

fig, axs = plt.subplots(nrows = 1, ncols = 4)

for indx, beta in enumerate(beta_list):
    w = np.zeros(n_trials)
    for loop in range(n_trials):
	    R = np.random.randn()*std + mean
	    w[loop] = w[loop-(loop>0)] + beta*(R - gamma*w[loop-(loop>0)])
    axs[indx].plot(w)
    axs[indx].set_title("beta = {}".format(beta))