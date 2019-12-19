#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:51:49 2019

@author: tom verguts
show likelihood curve for the learning model
"""

#%% initialize
import numpy as np
from ch6_generation import generate_learn
from ch6_likelihood import logL_learn
import matplotlib.pyplot as plt

learning_rate, temperature = 0.6, 1
ntrials = [100, 1000, 3000]
low_temp, high_temp, low_learn, high_learn, n_step = 0, +2, 0, +1, 20
x = np.linspace(low_learn, high_learn, n_step) # learning rate
y = np.linspace(low_temp,  high_temp,  n_step) # temperature 
X, Y = np.meshgrid(x, y)
v = np.ndarray((x.size, y.size, len(ntrials)))
filename = "simulation_data.csv"

#%% loop
for n_loop in np.arange(len(ntrials)):
    # generate some data
    generate_learn(alpha = learning_rate, beta = temperature, ntrials = ntrials[n_loop], file_name = filename)

    # calculate some likelihoods
    # calculate logL_learn over a grid of points
    for x_pos in np.arange(len(x)):
        for y_pos in np.arange(len(y)):
            x_step, y_step = x[x_pos], y[y_pos]
            v[x_pos, y_pos,n_loop] = logL_learn([x_step, y_step], ntrials[n_loop], file_name = filename)

#%% plot the grid
fig, axs = plt.subplots(1, len(ntrials))
ranges = []
for n_loop in np.arange(len(ntrials)):
    ranges.append(np.amax(v[:,:,n_loop]) - np.amin(v[:,:,n_loop]))
use_range = np.sort(ranges)[-1]

n_levels = 10
levels = np.ndarray((len(ntrials),n_levels))
for n_loop in np.arange(len(ntrials)):
    levels[n_loop,:] = np.linspace(np.mean(v[:,:,n_loop])-use_range/2, np.mean(v[:,:,n_loop])+use_range/2, num = n_levels)
    
            
for n_loop in np.arange(len(ntrials)):
    f0 = axs[n_loop].contour(X, Y,  v[:,:,n_loop], levels = levels[n_loop])
    loc = np.unravel_index(np.argmin(v[:,:,n_loop], axis=None), v[:,:,n_loop].shape) # maximum likelihood value
    f1 = axs[n_loop].contourf(X, Y, v[:,:,n_loop], levels = levels[n_loop], cmap="RdBu_r")
    axs[n_loop].clabel(f0, fontsize = 8)
    axs[n_loop].scatter(learning_rate, temperature, color = "k")
    axs[n_loop].scatter(x[loc[0]], y[loc[1]], marker = "X", color = "r") # plot maximumum likelihood
    axs[n_loop].set_title("n = {} trials".format(ntrials[n_loop]))
    if n_loop == 0:
        axs[n_loop].set_xlabel("learning rate")
        axs[n_loop].set_ylabel("temperature")    
    cb = fig.colorbar(f1, ax = axs[n_loop], shrink = 0.5)
    cb.ax.tick_params(labelsize=7)
fig.show()