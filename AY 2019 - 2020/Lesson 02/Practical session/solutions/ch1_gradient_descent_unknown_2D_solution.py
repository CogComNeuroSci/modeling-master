#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mehdisenoussi
2D extension by tomverguts
"""
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as pl
import numpy as np

def func(x1, x2):
    return  x1**2+ x2**2 + (x1/8)**2 - 3 * np.exp( - (x2**2) / 50) - 2*np.exp( -((x1+5)**2) / .2) * np.exp(-(x1**2)/2)

#%%

# create a array 'x' of npoints values from -10 to 10
npoints = 100
x1 = np.linspace(-10, 10, npoints)
x1 = np.tile(x1, (npoints, 1))
x2 = np.transpose(x1)

# compute f(x) for these values
y = func(x1, x2)

# create a figure with one subplot (1 row, 1 column)
fig = pl.figure()
ax = fig.add_subplot(111, projection = "3d")
# plot the function in the first plot
ax.plot_surface(x1, x2, y)


#%%
##############################
# Let's optimize algorithmically
##############################
# number of steps we do for the optimization
n_steps = 30
# scaling parameter
alpha = .1
# stepping parameter for gradient accuracy
delta = .1

# start from a random x and store this first value in an array 'x_grad'
x_grad = np.zeros((n_steps,2))
x_grad[0,:] = np.random.choice(x1[0,:], size = 2, replace = False)

# create an array 'y_grad' with the function's value for this random starting x
y_grad = np.zeros(n_steps)
y_grad[0] = func(x_grad[0,0], x_grad[0,1])

# plot the first point
col = "red"
ax.scatter(x_grad[0,0], x_grad[0,1], y_grad[0], 'o', color = col)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$f(x_1, x_2)$")

#%%
delta = 0.01 # step in input space that determines gradient accuracy
# optimization loop
ax.set_title("optimize in 2D")
for step in np.arange(1, n_steps):
    fig.suptitle('step: %i/%i' % (step, n_steps))
    x1_delta = x_grad[step-1,0] + delta
    x2_delta = x_grad[step-1,1] + delta
    y1_delta = func(x1_delta, x_grad[step-1,1])
    y2_delta = func(x_grad[step-1,0], x2_delta)
    # compute delta_x by using the derivative
    delta_x1 = -alpha * (y1_delta - y_grad[step-1]) / delta
    delta_x2 = -alpha * (y2_delta - y_grad[step-1]) / delta
    # update the new value of x in the array x_grad
    x_grad[step,0] = x_grad[step-1,0] + delta_x1
    x_grad[step,1] = x_grad[step-1,1] + delta_x2
    # update the new value of y in the array y_grad
    y_grad[step] = func(x_grad[step,0], x_grad[step,1])
    # plot these latest values on the function plot
    ax.scatter(x_grad[step,0], x_grad[step,1], y_grad[step], 'o', color = col)

    # to refresh the figure
    fig.canvas.draw()
    # wait before you go to the next iteration of the loop
    fig.waitforbuttonpress(.1)


fig.suptitle('End of the optimization!')


