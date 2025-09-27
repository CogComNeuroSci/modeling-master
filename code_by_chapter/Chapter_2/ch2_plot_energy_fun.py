#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pieter Huycke, Mehdi Senoussi

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
This script plots the pet detector's energy function in 2D (equation (2.3))

Inspecting the plot, you can see that if you start your descent on the "cat" side,
you will end up at a cat output ((y_cat, y_dog) = (10, 0));
and if you start at the "dog" side, you end up at a dog output ((y_cat, y_dog) = (0, 10)).
However, if the competition becomes too weak (abs(w) is small), then the minimum
of the energy function goes to (10, 10) (at a grid bounded by 10)

A tendency toward cat or dog can be implemented by multiplying the X or Y terms with a constant in Z = - X - Y - w*x*Y
Such a constant can be thought of as a drift rate for resp. cats or dogs (called in_cat and in_dog in the MCP book)
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import ticker

# the resolution of our plot
n_unitsteps = 50

# create the different value that the first dimension (let's call it x) will take
xs = np.linspace(-5, 10, n_unitsteps)
# create the different value that the second dimension (let's call it y) will take
ys = np.linspace(-5, 10, n_unitsteps)
# create a 2D array representing all the possible combinations of x and y
X, Y = np.meshgrid(xs, ys)

# the weight between the cat and dog units
w = -1
# the energy function
Z = - 10*X - Y - w*X*Y

# plot the energy as a function of all the x and y combinations
fig = plt.figure()
#ax = fig.gca()    # in older matplotlib versions, use this instead of next line
ax = fig.add_subplot(111, projection = "3d")
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.02f'))

ax.set_xlabel('$y_{cat}$')
ax.set_ylabel('$y_{dog}$')
ax.set_zlabel('$Energy$')


