#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mehdisenoussi
"""

from matplotlib import pyplot as plt
import numpy as np


def func1(x):
    # implement func1 (see the example function in the chapter 1)
    return (x-1)**2

def der_func1(x):
    # the derivative of func1
    return 2*(x-1)


# create a array 'x' of 500 values from -10 to 10
x = np.linspace(-10, 10, 500)
# compute f(x) for these values
y = func1(x)
# compute f'(x) for these values
y_der = der_func1(x)

# create a figure with two subplots (2 rows, 1 column)
fig, axes = plt.subplots(nrows = 2, ncols = 1)
# plot the function in the first plot
axes[0].plot(x, y, color = 'k')
# add a grid (just for vizualization)
axes[0].grid(True)
# plot the derivative in the second plot
axes[1].plot(x, y_der, color = 'k')
axes[1].grid(True)

# number of steps we do for the optimization
n_steps = 20
# scaling parameter
alpha = .05

##############################
# Let's do it algorithmically
##############################
# start from a random x and store this first value in an array 'x_grad'
x_grad = np.zeros(n_steps)
x_grad[0] = np.random.choice(x, size = 1, replace = False)
# create a list 'y_grad' with the function's value for this random starting x
y_grad = np.zeros(n_steps)
y_grad[0] = func1(x_grad[0])

# optimization loop
for step_i in np.arange(1, n_steps):
    axes[0].set_title('step: {0}/{1}'.format(step_i, n_steps))
    # compute delta_x using the derivative
    delta_x = -alpha * der_func1(x_grad[step_i-1])
    # update the new value of x in the list x_grad
    x_grad[step_i] = x_grad[step_i-1] + delta_x
    # update the new value of y in the list y_grad
    y_grad[step_i] = func1(x_grad[step_i])
    # plot these latest values on the function plot
    axes[0].plot(x_grad[step_i], y_grad[step_i], 'bo')
    # plot these latest values on the derivative's plot
    axes[1].plot(x_grad[step_i], der_func1(x_grad[step_i]), 'bo')
    # to refresh the figure
    fig.canvas.draw()
    # to wait for any button press to go to the next iteration of the loop
    # you can make this "automatic" by changing the 0 to a number of seconds
    fig.waitforbuttonpress(0)


axes[0].set_title('End of the optimization!')

