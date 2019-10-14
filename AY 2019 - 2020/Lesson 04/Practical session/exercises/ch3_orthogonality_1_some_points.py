#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mehdisenoussi
"""

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as pl


# learning parameter
beta = .4

#############################
# Some points

fig = pl.figure(figsize = (13, 7))
axs = [];
axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
axs.append(fig.add_subplot(1, 2, 2))

# Coordinates of our first point
x1_1 = [1]
x2_1 = [0]
x3_1 = [0]

# Coordinates of our second point
x1_2 = [0]
x2_2 = [0]
x3_2 = [1]

axs[0].plot3D(x1_1, x2_1, x3_1, 'ro', label='cat')
axs[0].plot3D(x1_2, x2_2, x3_2, 'bo', label='dog')
axs[0].legend()

axs[0].zaxis.set_major_locator(LinearLocator(10))
axs[0].zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

axs[0].set_xlabel('$x_{1}$: Has its pic. on FB')
axs[0].set_ylabel('$x_{2}$: Has 4 legs')
axs[0].set_zlabel('$x_{3}$: Bites visitors')

# y1 and y2
weights = np.zeros(shape = [2, 3])
targets = np.array([[1, 0], [0, 1]])

weights = weights + beta * np.dot(targets[0,:][:, np.newaxis], np.array([x1_1, x2_1, x3_1]).T)
weights = weights + beta * np.dot(targets[1,:][:, np.newaxis], np.array([x1_2, x2_2, x3_2]).T)

axs[1].plot([0, 1], [0,1], 'k--')
axs[1].set_xlabel('$y_{1}:$ cat detector')
axs[1].set_ylabel('$y_{2}:$ dog detector')

x1_new = [1]
x2_new = [1]
x3_new = [0]
activations = np.dot(weights, np.array([x1_new, x2_new, x3_new]))
axs[1].plot(activations, 'ro')

x1_new = [0]
x2_new = [1]
x3_new = [1]
activations = np.dot(weights, np.array([x1_new, x2_new, x3_new]))
axs[1].plot(activations[0],activations[1], 'bo')
