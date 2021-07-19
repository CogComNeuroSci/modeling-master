#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July 19, 2021

@author: tom verguts
TF-based hopfield network for arbitrary vectors X and Y
Both X, Y, -X, and -Y are attractors in this case.
Note also distance(X, -X) = 20 (in case N = 100)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define fixed parameters
N = 100
X = 2*(np.random.random(N)>0.5)-1 # +1 / -1 coding
X = X[:,np.newaxis]
Y = 2*(np.random.random(N)>0.5)-1 # +1 / -1 coding
Y = Y[:, np.newaxis]

w    = np.matmul(X, X.T) + np.matmul(Y, Y.T)
threshold = 0
b    = np.array(N*[threshold])
b    = b[:, np.newaxis]
start_pattern = (2*np.random.random(N)>0.5)-1
start_pattern = start_pattern[:, np.newaxis]

n_sample = 50
distance = np.zeros((2, n_sample))

# a function to sample the network iteratively
def hopfield(start_pattern = None, n_sample = 0):
	pattern = tf.cast(start_pattern, dtype = tf.int64)
	for loop in range(n_sample):
		distance[:,loop] = [np.linalg.norm(X-pattern), np.linalg.norm(Y-pattern)]
		net_input = tf.matmul(w, pattern) #+ tf.multiply(pattern, b)
		clipped   = tf.cast(tf.math.greater(net_input, 0), tf.float32)
		pattern   = clipped*2 - 1
		pattern = tf.cast(pattern, dtype = tf.int64)
	return pattern

pattern = hopfield(start_pattern = start_pattern, n_sample = n_sample)
#print(pattern.numpy())
print("distance from X: ", np.linalg.norm(X-pattern))
print("distance from Y: ", np.linalg.norm(Y-pattern))
plt.plot(distance[0,:], color = "red")
plt.plot(distance[1,:], color = "black")