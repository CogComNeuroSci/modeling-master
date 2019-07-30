#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:49:30 2019

@author: tom verguts
plotting dots for creating exam questions
"""

import matplotlib.pyplot as plt
#import numpy as np

dots_C1 = [ [(1, 1.5),  (1, 0.5), (2, 0.5)], \
            [(0.5, 1.5), (0.5, 2)], \
            [(0.2, 1), (1, 0.2)] ]
dots_C2 = [ [(2, 2), (0.5, 2), (1, 2.3)], \
            [(1.5, 0.5), (1, 1)], \
            [(0.5, 0.5), (2, 2)] ]
marker_colors = ["red", "green"]
titles = ["A", "B", "C"]

fig, axs = plt.subplots(nrows = 1, ncols = len(dots_C1))
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
for loop in range(len(dots_C1)):
    axs[loop].set_xlim([0, 3])
    axs[loop].set_ylim([0, 3])
    axs[loop].set_title(titles[loop])
    for dots_loop in range(len(dots_C1[loop])):
        axs[loop].plot(dots_C1[loop][dots_loop][0], dots_C1[loop][dots_loop][1], \
               marker = "o", markerfacecolor = marker_colors[0], markersize = 12, markeredgecolor = "black")
 
    for dots_loop in range(len(dots_C2[loop])):
        axs[loop].plot(dots_C2[loop][dots_loop][0], dots_C2[loop][dots_loop][1], \
               marker = "o", markerfacecolor = marker_colors[1], markersize = 12, markeredgecolor = "black")