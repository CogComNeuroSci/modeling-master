#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:38:59 2021

@author: tom verguts
1-dimensional function minimization with TensorFlow2
"""

import tensorflow as tf
from tensorflow.python.training import gradient_descent
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

plot_results = True # plot the process: later points are lighter
if plot_results:
    wait_for_press = True # warning: if set to True, don't plot to console
    mpl.use("Qt5Agg")     # this should plot the figure outside the console
    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba(0)
    fig, ax = plt.subplots()

step_size, n_steps = 0.1, 10
x = tf.Variable(initial_value = 10.0, trainable = True)


def f_x():
    return (x - 5)**2

func_to_use = f_x

for step in range(n_steps):
    print("x = {:.2f}, f(x) = {:.2f}".format(x.numpy(), func_to_use().numpy()))
    gradient_descent.GradientDescentOptimizer(step_size).minimize(f_x) # core of the code
    if plot_results:
        col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba((step+2)/float(n_steps+2))
        a = ax.plot(x.numpy(), func_to_use().numpy(), 'o', mec = 'k', color = col)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        fig.canvas.draw()
        if wait_for_press:
            plt.title('step: {}/{}'.format(step+1, n_steps))
            fig.waitforbuttonpress(0) # if you want to step through the function

if plot_results:
    plt.title('Optimization done.')			
		