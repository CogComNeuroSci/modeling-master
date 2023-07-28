#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 10:03:12 2018

@author: tom verguts
chapter 11: illustration of the basic bayesian principle
that posterior distribution gradually sharpens as data accumulate;
the prior strength (informativeness) determines how quickly one can 'escape'
from the prior
"""

#%% import and initialize
import numpy as np
import matplotlib.pyplot as plt

low, high = 0, 1
p = 0.8
sample_size = [0, 50, 500, 50]
n_support = 500
v = np.linspace(low, high, num = n_support) # vector of points where density is calculated
delta_p = 1/n_support
signif = 0.05/2

#%% figure 11.1: effect of different amounts of data on posterior (plots a-c) and effect of prior (plot d)
alpha, beta = [2, 2, 2, 2], [2, 2, 2, 40] # prior parameters for the beta distribution

fig, axes = plt.subplots(nrows = 2, ncols = 2)
for index in range(4):
    r, c = index//2, index%2
    n_heads = np.random.binomial(n = sample_size[index], p = p) # the data
    dens = (v**(alpha[index]-1+n_heads))*((1-v)**(beta[index]-1+(sample_size[index]-n_heads)))
    tot = np.sum(dens)*delta_p
    dens = dens/tot # normalize to integral of 1
    mean_theta = np.dot(dens, v)*delta_p
    max_theta = v[np.argmax(dens)]
    left_point  = np.argmin(np.abs(np.cumsum(dens*delta_p)-signif))      # left point of confidence bar
    right_point = np.argmin(np.abs(np.cumsum(dens*delta_p)-(1-signif)))  # right point of confidence bar
    if (index==0) or (index==2):
        axes[r, c].set_ylabel("posterior belief in p")
    if (index==2) or (index==3):
        axes[r, c].set_xlabel("p")    
    axes[r, c].axis([low, high, 0, 25])
    axes[r, c].plot(v, dens, color = "black")
    axes[r, c].plot(
      v[left_point:right_point],np.zeros(right_point-left_point)+0.15, color = "black", linewidth = 7)
    axes[r, c].set_title("sample size = {}".format(sample_size[index]))
    axes[r, c].plot(mean_theta, 2, marker = "o", color = "black")
    axes[r, c].plot(max_theta, 2,  marker = "x", color = "black")

#%%figure 11.2: sequential updating of the posterior
fig, axes = plt.subplots(nrows = 2, ncols = 2)
n_trial = 1000
plot_set = [1, 9, 19, 499] # after these sample sizes, a plot is drawn; change to lower (higher) numbers if you want to see the posterior at earlier (later) stage in sampling
alpha, beta = 2, 2
prior = (v**(alpha-1))*((1-v)**(beta-1))
plot_label = 0
for trial_loop in range(n_trial):
    # sample 
    x = np.random.binomial(n = 1, p = p)
    post = prior * ((v**x) * ((1-v)**(1-x)))
    post = post/(np.sum(post)*delta_p) # normalize the posterior
    prior = post                       # today's posterior is tomorrow's prior 
    if trial_loop in plot_set:         # plot the current-day posterior
        r, c = plot_label//2, plot_label%2
        axes[r, c].plot(v, post, color = "black")
        axes[r, c].set_ylim(0, 30)
        axes[r, c].set_title("posterior after {} trials".format(trial_loop+1))
        plot_label += 1    