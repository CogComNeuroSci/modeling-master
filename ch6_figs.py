#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2, 2018

@author: tom verguts
pics from chapter 6
"""

import numpy as np
import matplotlib.pyplot as plt

#%% fig 6.2
x = np.linspace(-3, 3, 50)
y1 = x**2
y2 = np.log(y1)

fig, axs = plt.subplots(nrows = 1, ncols = 2)

axs[0].plot(x, y1, color = "black")
axs[0].set_title("y = x**2")
axs[1].plot(x, y2, color = "black")
axs[1].set_title("y = log(x**2)")

#%% fig 6.3
logL_range = 100
n = [[7, 3], [70, 30]]
fig, axs = plt.subplots(nrows = 1, ncols = 2)

p = np.linspace(0+1/50, 1-1/50, 50)
for loop in range(2):
    logL = n[loop][0]*np.log(p) + n[loop][1]*np.log(1-p)
    av = np.mean(logL)
    axs[loop].plot(p, logL, color = "black")
    axs[loop].set_ylim(av-logL_range/2, av+logL_range/2)
    axs[loop].set_title("{} data points".format(n[loop][0]+n[loop][1]))
