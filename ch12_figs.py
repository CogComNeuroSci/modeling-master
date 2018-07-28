#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 10:03:12 2018

@author: tom
chapter 12: basic bayes
"""

import numpy as np
import matplotlib.pyplot as plt

low, high = 0, 1
p = 0.8
sample_size = [5, 50, 500, 50]
n_support = 500
alpha, beta = [2, 2, 2, 2], [2, 2, 2, 40] # prior parameters for the beta distribution
v = np.linspace(low, high, num = n_support) # vector of points where density is calculated
delta_p = 1/n_support
signif = 0.05/2

fig, axes = plt.subplots(nrows = 2, ncols = 2)
#plt.subplot(121)
for index in range(4):
    r, c = index//2, index%2
    n_heads = np.random.binomial(sample_size[index], p)
    dens = (v**(alpha[index]-1+n_heads))*((1-v)**(beta[index]-1+(sample_size[index]-n_heads)))
    tot = np.sum(dens)*delta_p
    dens = dens/tot # normalize to integral of 1
    left_point =  np.argmin(np.abs(np.cumsum(dens*delta_p)-signif))
    right_point = np.argmin(np.abs(np.cumsum(dens*delta_p)-(1-signif)))
    if (index==0) or (index==2):
        axes[r, c].set_ylabel("posterior belief in p")
    if (index==2) or (index==3):
        axes[r, c].set_xlabel("p")    
    axes[r, c].axis([low, high, 0, 25])
    axes[r, c].plot(v, dens, color = "black")
    axes[r, c].plot(
      v[left_point:right_point],np.zeros(right_point-left_point)+0.15, color = "red", linewidth = 7)
    axes[r, c].set_title("sample size = {}".format(sample_size[index]))
    