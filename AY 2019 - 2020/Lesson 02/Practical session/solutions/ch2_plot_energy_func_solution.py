#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pieter Huycke, Mehdi Senoussi

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
This script plots the pet detector's energy function in 2D

"""
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import ticker

# the resolution of our plot
n_unitsteps = 50

# create the different value that the first dimension (let's call it x) will take
xs = np.linspace(0, 10, n_unitsteps)
# create the different value that the second dimension (let's call it y) will take
ys = np.linspace(0, 10, n_unitsteps)
# create a 2D array representing all the possible combinations of x and y
X, Y = np.meshgrid(xs, ys)

# the weight between the cat and dog units
w = -.2
# the energy function
Z = - X - Y - w*X*Y

# plot the energy as a function of all the x and y combinations
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.02f'))

ax.set_xlabel('$x_{cat}$')
ax.set_ylabel('$y_{dog}$')
ax.set_zlabel('$Energy$')


