#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts
Does cats-dogs network updating via minimization of cost function (2.3)
it thus implements the model defined in (2.4) and (2.5)
"""

import tensorflow as tf
from tensorflow.python.training import gradient_descent
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# initialize variables
x_cat = np.array([1, 1, 0]) # prototypical cat
x_dog = np.array([0, 1, 1]) # prototypical dog
x = x_dog

W = np.array([[1, 0.5, 0.1], [0.1, 0.5, 1]])
net = np.matmul(W, x).reshape(2,1) # net input to the cat and dog output units
w_inh = -0.4    # lateral inhibition between cat and dog
W_inh = w_inh*np.array([[0, 1], [1, 0]])
step_size = 0.05
n_steps = 100
y = np.ndarray((n_steps, 2))
noise = 1

# define TensorFlow components
Y  = tf.Variable(initial_value = np.random.randn(1, 2))

# define functions
def cost():
    # this is just the cost function 2.3 in matmul notation
	return -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.transpose(Y)) 

def plot_activation():
    # plot the cat / dog competition
    ax.plot(range(n_steps), y[:,0], color = "red", label = "cat")
    ax.plot(range(n_steps), y[:,1], color = "black", label = "dog")        
    ax.set_xlabel("time")
    ax.set_ylabel("activation")
    ax.legend()

# start the optimization	
for step in range(n_steps):
    Y.assign(Y + np.random.rand(1,2)*noise) # add some noise
    y[step] = Y.numpy()                     # store it as numpy array for plotting purposes
    gradient_descent.GradientDescentOptimizer(step_size).minimize(cost) # core of the code
    Y.assign(tf.math.maximum(0, Y))     # clip it at zero

plot_activation()