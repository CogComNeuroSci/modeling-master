#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:38:59 2021

@author: tom verguts
1-dimensional function minimization with TensorFlow2
a**x means a raised to the power x
this code does the same as ch1_tf2_simple_min
but plots only after the computation rather than during
"""

import tensorflow as tf
from tensorflow.python.training import gradient_descent
import matplotlib.pyplot as plt
import numpy as np

plot_results = True
if plot_results:
	fig, ax = plt.subplots(nrows = 1, ncols = 1)
	ax.set_xlim([-5, 5]) # make these limits appropriate for your function if you want to see something happening
	ax.set_ylim([-2, +10])     # same here
	ax.set_xlabel("x")
	ax.set_ylabel("f(x)")

step_size, n_steps = 0.01, 100
x = tf.Variable(initial_value = 0.0, trainable = True)

def f_x():
    return (x - 5)**2

def my_function_x():
    return x**2 +7

def my_other_function_x():
    return - (-x**2 +7)

def f2_x():
	return 2*(x**4) -0.3*(x**3) - 2.5*(x**2)  

func_to_use = f_x

data = np.ndarray((n_steps, 2)) # we collect the data here for plotting afterwards

for step in range(n_steps):
	gradient_descent.GradientDescentOptimizer(step_size).minimize(func_to_use) # core of the code	
	data[step, 0] = x.numpy()
	data[step, 1] = func_to_use().numpy()
	print("x = {:.2f}, f(x) = {:.2f}".format(data[step, 0], data[step, 1]))

if plot_results:
	ax.plot(data[:, 0], data[:, 1], marker = "o")
	ax.set_title('optimization results')    