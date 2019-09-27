#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mehdisenoussi
"""
import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import cm
import matplotlib.pyplot as pl
import numpy as np

# you can use any of these two functions, to use one just comment the other and
# run the script.
def func(x):
    return  (x/8)**2 - 3 * np.exp( - (x**2) / 50) - 2*np.exp( -((x+5)**2) / .2)


def func(x):
    return 5/(.7*x**3 + 10*np.sin(x)) + 10*np.cos(x) + .5*x**2


# for plotting points with different colors depending on the order
norm = mpl.colors.Normalize(vmin = 0, vmax = 1)


# create a array 'x' of 500 values from -10 to 10
x = ...
# compute f(x) for these values
y = func(x)

# create a figure with two subplots (1 row, 1 column)
fig, axes = ...
# plot the function in the first plot
axes.plot(..., color = 'k')
# add a grid
axes.grid(True)

# number of steps we do for the optimization
n_steps = 30
# scaling parameter
alpha = .1


##############################
# Let's do it algorithmically
##############################
# start from a random x and store this first value in an array 'x_grad'
x_grad = np.zeros(n_steps)
x_grad[0] = np.random.choice(x, size = 1, replace = False)

# create an array 'y_grad' with the function's value for this random starting x
y_grad = ...
y_grad[0] = func(...)

# plot the first point
col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba(0)
axes.plot(..., ..., 'o', mec = 'k', color = col)

# get a second point near the first random one to compute the slope at that value of x
x_grad[1] = ...
# fill in 'y_grad' for this second point
y_grad[1] = ...

# plot the second point
col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba(1./(n_steps+2))
axes.plot(..., ..., 'o', mec = 'k', color = col)


# optimization loop
for step_i in np.arange(2, n_steps):
    axes.set_title('step: %i/%i' % (step_i+2, n_steps+1))
    # compute delta_x using by computing the derivative
    delta_x = ...
    # update the new value of x in the array x_grad
    x_grad[step_i] = ...
    # update the new value of y in the array y_grad
    y_grad[step_i] = ...
    # plot these latest values on the function plot
    col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba((step_i+2.)/(n_steps+2))
    axes.plot(..., ..., 'o', mec = 'k', color = col)

    # to refresh the figure
    fig.canvas.draw()
    # wait before you go to the next iteration of the loop
    fig.waitforbuttonpress(.1)


axes.set_title('End of the optimization!')


