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


def func1(x):
    # implement func1 (see the example function in chapter 1)
    return ...

def der_func1(x):
    # the derivative of func1
    return ...



# create a array 'x' of 500 values from -10 to 10
x = ...
# compute f(x) for these values
y = func1(x)
# compute f'(x) for these values
y_der = der_func1(x)

# create a figure with two subplots (2 rows, 1 column)
fig, axes = ...
# plot the function in the first plot
axes[0].plot(..., color = 'k')
# add a grid
axes[0].grid(True)
# plot the derivative in the second plot
axes[1].plot(..., color = 'k')
axes[1].grid(True)


# number of steps we do for the optimization
n_steps = 20
# scaling parameter
alpha = ...


##############################
# Let's do it algorithmically
##############################
# start from a random x and store this first value in an array 'x_grad'
x_grad = np.zeros(n_steps)
x_grad[0] = np.random.choice(x, size = 1, replace = False)

# create an array of zeros 'y_grad' with the function's value for this random starting x
y_grad = np.zeros(n_steps)
y_grad[0] = func1(...)

# optimization loop
for step_i in np.arange(n_steps):
    axes[0].set_title('step: %i/%i' % (step_i, n_steps+1))
    # compute delta_x using the derivative
    delta_x = ...
    # update the new value of x in the array x_grad
    x_grad[step_i] = ...
    # update the new value of y in the array y_grad
    y_grad[step_i] = ...
    # plot these latest values on the function plot
    axes[0].plot(..., ..., 'bo')
    # plot these latest values on the derivative's plot
    axes[1].plot(..., ..., 'bo')
    # to refresh the figure
    fig.canvas.draw()
    # to wait for any button press to go to the next iteration of the loop
    # you can make this "automatic" by changing the 0 to a number of seconds
    fig.waitforbuttonpress(0)

# print a title to indicate that we've finished the optimization
axes[0].set_title('End of the optimization!')
