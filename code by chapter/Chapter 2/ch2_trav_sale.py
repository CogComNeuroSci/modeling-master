#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:49:42 2021

@author: tom verguts
The travelling salesperson (idea after Rojas, 1996; originally proposed by Hopfield & Tank)
"""

#%% import
import numpy  as np
import tensorflow as tf

#%% initialize
# dij is distance between city i and city j
d12, d13,  d23 = 1, 4, 4
d = np.array([[0, d12, d13], [d12, 0, d23], [d13, d23, 0]]) # distances
dim = d.shape[0]
g = 2 # sum constraint; this ensures that each city is visited once, and that at each time step, only one city is visited

W = np.zeros((dim**2, dim**2), dtype = np.double)
W =  np.kron(np.ones((dim, dim)), np.eye(dim, dim))
W += np.kron(np.eye(dim, dim),np.ones((dim, dim)))
W -= 2*np.eye(9)
W *= g
#W += np.kron(np.ones((dim, dim)), d)
W = -W
threshold = -g/2
print(W)

# a function to sample the network iteratively
def hopfield(start_pattern = None, n_sample = 0):
	pattern = tf.cast(start_pattern, dtype = tf.double)
	for loop in range(n_sample):
		net_input = tf.matmul(W, pattern) - tf.multiply(pattern, threshold)
		clipped   = tf.cast(tf.math.greater(net_input, 0), tf.double)
		pattern   = clipped
	return pattern

#%% run algorithm
#start_pattern = (np.random.random(9)>0.5)*1 # start in random state
start_pattern = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
start_pattern = start_pattern[:, np.newaxis]
print(start_pattern)

pattern = hopfield(start_pattern = start_pattern, n_sample = 10)
pattern = np.array(pattern)
print(pattern)
path = np.ndarray(3)
for loop in range(3):
	v = pattern[3*loop + np.array(range(3))]
	path[loop] = np.argmax(v)
print("path: ",path)
print("energy: ",-(np.matmul(np.matmul(pattern.T,W),pattern)/2 - np.sum(pattern)*threshold))