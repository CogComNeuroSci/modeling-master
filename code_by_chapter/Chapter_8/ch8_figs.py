#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:28:40 2018

@author: tom verguts
n-armed bandit
solved with gradient ascent (chap 8) with different values of gamma;
see eq 8.1 for definition of gamma (inverse temperature)
this generates fig 8.2
"""
#%% imports and initializations
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision = 2)
n_simulations = 20
n_trials = 500
beta = [0.5]*4            # the learning rate
#beta = [0.01, .8, 5, 10] # in case different bandits have different learning rates
p = [0.2, 0.4, 0.6, 0.8]  # payoff probabilities
gamma = [0.01, .8, 5, 10] # inv temperatures
#gamma = [1]*4
window_conv = 20  # window size for convolution; plots will be smoother with larger window_conv
color_list = ["black", "black", "black", "black"]
label_list = ["a)", "b)", "c)", "d)"]
xlabels, ylabels = [2, 3], [0, 2]
fig, axs = plt.subplots(nrows = 2, ncols = 2)
r_tot = np.zeros((len(gamma),n_trials))

#%% let's play: different gamma's
for simulation_loop in range(n_simulations):
    for gamma_loop in range(len(gamma)):
        w = np.random.random(len(p)) # can be interpreted as weights or Q-values
        r= []
        for loop in range(n_trials):
            prob = np.exp(gamma[gamma_loop]*w)
            prob = prob/np.sum(prob)
            choice = np.random.choice(range(len(p)),p=prob)
            r.append(np.random.choice([0,1],p=[1-p[choice], p[choice]]))
			# now follows the key equation (8.5):
            w += np.asarray((range(len(p))==choice)-prob)*beta[gamma_loop]*gamma[gamma_loop]*r[-1]
        r_tot[gamma_loop,:] = r_tot[gamma_loop,:] + r            
r_tot = r_tot/n_simulations

#%% plotting results
for gamma_loop in range(len(gamma)):
    v = np.convolve(r_tot[gamma_loop,:],np.ones(window_conv)/window_conv)
    row, col = int(np.floor(gamma_loop/2)), gamma_loop%2
    axs[row, col].plot(v[window_conv:-window_conv], color = color_list[gamma_loop])
    axs[row, col].set_title("gamma = {}".format(gamma[gamma_loop]))
    axs[row, col].set_title(label_list[gamma_loop])
    axs[row, col].set_ylim(bottom=0, top=1)
    if gamma_loop in xlabels:
        axs[int(np.floor(gamma_loop/2)), gamma_loop%2].set_xlabel("trial nr")
    if gamma_loop in ylabels:
        axs[int(np.floor(gamma_loop/2)), gamma_loop%2].set_ylabel("average reward")
          