#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:19:06 2019

@author: tom verguts
pics for chapter 7, testing computational model
"""

import numpy as np
import matplotlib.pyplot as plt

dep = [0, 0.8, 1] # trial-by-trial dependencies
n_trials = 100
#titles = ["Fig 7.1a: No dependency", "Fig 7.1b: Strong dependency"]

fig, axs = plt.subplots(nrows = 1, ncols = 3)
x = range(n_trials)
y = np.empty(n_trials)

for plot_loop in range(3):
    for step in range(n_trials-1):
        y[step+1] = dep[plot_loop]*y[step] + np.random.randn()*0.01
    y_hat = y - np.mean(y)    # the residuals
    axs[plot_loop].scatter(x, y_hat, color = "black")
#    axs[plot_loop].set_title(titles[plot_loop])
    axs[plot_loop].set_xlabel("Trial nr")