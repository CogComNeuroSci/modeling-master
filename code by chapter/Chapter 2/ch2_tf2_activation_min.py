#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts
Does activation updating via minimization of a cost function (equation 2.3)
It's also possible to change Y explicitly, rather than implicitly via a cost function, as done here
In particular, one can explicitly code equations like (2.4) and (2.5)
"""

import tensorflow as tf
from tensorflow.python.training import gradient_descent
import numpy as np
import matplotlib.pyplot as plt


# initialize variables
x = np.array([1, 1, 0]) # input units
W = np.array([[2, 0, 0], [0, 0, 2]]).astype(np.float32) # transformation matrix x to net input
net = np.matmul(W, x).reshape(2,1).astype(np.float32)   # matrix multiplication (matmul)
w_inh = -0.5 # inhibition between output (Y) units
W_inh = w_inh*np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
n_steps = 50
activation = np.ndarray((n_steps,2))
step_size = 0.1

Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32)/10, name="Y") # this will be optimized to minimize cost

def cost(): # this is the cost function (2.3) (in slightly more general notation than in the MCP book)
	return -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.transpose(Y)) 


for i in range(n_steps):
	Y.assign(Y + np.random.rand(1,2)*5)
	activation[i] = Y.numpy()
	opt = gradient_descent.GradientDescentOptimizer(step_size).minimize(cost)
	Y.assign(tf.math.maximum(0, Y))

	
plt.plot(activation[:,0], color = "red")
plt.plot(activation[:,1], color = "black")        