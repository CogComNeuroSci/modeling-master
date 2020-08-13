#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:28:40 2018

@author: tom verguts
n-armed bandit
solved with 
- gradient ascent (chap 9) with different values of gamma
"""
#%% come in
import numpy as np
import matplotlib.pyplot as plt

#%% for starters
np.set_printoptions(precision=2)
n_simulations = 20
ntrial = 500
alpha = [0.5]*4
#alpha = [0.01, .8, 5, 10]
beta = 2
p = [0.2, 0.4, 0.6, 0.8] # payoff probabilities
gamma = [0.01, .8, 5, 10]
#gamma = [1]*4
window = 30
window_conv = 20
threshold = 0.8
color_list = ["black", "black", "black", "black"]
fig, axs = plt.subplots(nrows=2, ncols=2)
r_tot = np.zeros((len(gamma),ntrial))

#%% let's play: different gamma's
for simulation_loop in range(n_simulations):
    for gamma_loop in range(len(gamma)):
        w = np.random.random(len(p)) # can be interpreted as weights or Q-values
        r= []
        for loop in range(ntrial):
            prob = np.exp(gamma[gamma_loop]*w)
            prob = prob/np.sum(prob)
            choice = np.random.choice(range(len(p)),p=prob)
            r.append(np.random.choice([0,1],p=[1-p[choice], p[choice]]))
            w += np.asarray((range(len(p))==choice)-prob)*alpha[gamma_loop]*gamma[gamma_loop]*r[-1]
        r_tot[gamma_loop,:] = r_tot[gamma_loop,:] + r            
r_tot = r_tot/n_simulations

#%% printing & plotting

for gamma_loop in range(len(gamma)):
    v = np.convolve(r_tot[gamma_loop,:],np.ones(window_conv)/window_conv)
    #v = r_tot[gamma_loop,:]
    axs[int(np.floor(gamma_loop/2)), gamma_loop%2].plot(v[window_conv:-window_conv], color = color_list[gamma_loop])
    axs[int(np.floor(gamma_loop/2)), gamma_loop%2].set_title("gamma = {}".format(gamma[gamma_loop]))
    axs[int(np.floor(gamma_loop/2)), gamma_loop%2].set_ylim(bottom=0, top=1)