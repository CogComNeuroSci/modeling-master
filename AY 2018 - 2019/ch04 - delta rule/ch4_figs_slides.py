#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom verguts
create plots for slides chapter 4, the delta rule
"""
import matplotlib.pyplot as plt
import numpy as np

# plot some vectors
nplot = 3
fig, axs = plt.subplots(nrows=1,ncols=nplot)
v = np.array([[1, 1], [-1, 1], [1, 0]]) # collection of all vectors (not of length unity)
to_be_plotted = [[0, 1], [0, 2], [0, 1, 2]]
for plot_loop in range(nplot):
    for dot_loop in to_be_plotted[plot_loop]:
        axs[plot_loop].scatter(v[dot_loop][0], v[dot_loop][1], c = "black")
    axs[plot_loop].set_xlim(-1.5, 1.5)
    axs[plot_loop].set_ylim(-1.5, 1.5)
    axs[plot_loop].set_xlabel("Input dimension 1")
    axs[plot_loop].set_ylabel("Input dimension 2")
    