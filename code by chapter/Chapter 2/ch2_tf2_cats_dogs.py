#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts
Does cats-dogs network updating via minimization of activation function (2.3)
"""

import tensorflow as tf
from tensorflow.python.training import gradient_descent
import numpy as np
import matplotlib.pyplot as plt

# initialize variables
#x_cat = np.array([1, 1, 0]) # prototypical cat
x_dog = np.array([0, 1, 1]) # prototypical dog
x = x_dog
W = np.array([[2, 1, 0], [0, 1, 2]]).astype(np.float32)
net = np.matmul(W, x).reshape(2,1).astype(np.float32) # net input to the cat and dog output units
w_inh = -0.4    # lateral inhibition between cat and dog
W_inh = w_inh*np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
step_size = 0.05
n_steps = 100
y = np.ndarray((n_steps, 2))

# define TensorFlow components
Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32), name="Y")


def cost():
	return -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.transpose(Y)) 


for i in range(n_steps):
	Y.assign(Y + np.random.rand(1,2)*5)
	y[i] = Y.numpy()
	opt = gradient_descent.GradientDescentOptimizer(step_size).minimize(cost)
	Y.assign(tf.math.maximum(0, Y))

# plot the cat / dog competition
plt.plot(range(n_steps), y[:,0], color = "k") # the cat
plt.plot(range(n_steps), y[:,1], color = "r") # the dog        