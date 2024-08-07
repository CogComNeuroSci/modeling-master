#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:49:42 2021

@author: tom verguts
Script to implement the rooks problem or
the travelling salesperson (idea after Rojas, 1996; originally proposed by Hopfield & Tank)
"""

#%% import
import numpy  as np
import tensorflow as tf

#%% initialize
# dij is distance between city i and city j
# 2 cities
#d12 = 1
#d = np.array([[0, d12], [d12, 0]]) # distances
# 3 cities
# d12, d13,  d23 = 1, 1, 4
# d = np.array([[0, d12, d13], [d12, 0, d23], [d13, d23, 0]]) # distances
# 4 cities
d12, d13, d14, d23, d24, d34 = 1, 1, 1, 1, 2, 2
d = np.array([[0, d12, d13, d14], [d12, 0, d23, d24], [d13, d23, 0, d34], [d14, d24, d34, 0]]) # distances

dim = d.shape[0]
g = 1 # sum constraint; this ensures that each city is visited once, and that at each time step, only one city is visited

W = np.zeros((dim**2, dim**2), dtype = np.double)
W =  np.kron(np.ones((dim, dim)), np.eye(dim, dim))
W += np.kron(np.eye(dim, dim),np.ones((dim, dim)))
W -= 2*np.eye(dim**2)
W *= g
#W += np.kron(np.ones((dim, dim)), d) # if switched off, it's w/o distances; making it a rooks problem
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
start_pattern = (np.random.random(dim**2)>0.5)*1 # start in random state
#start_pattern = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])  # start at chosen location
start_pattern = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])  # start at chosen location
start_pattern = start_pattern[:, np.newaxis]

pattern = hopfield(start_pattern = start_pattern, n_sample = 10)
pattern = np.array(pattern)
print(pattern)
path = np.ndarray(dim)
for loop in range(dim):
	v = pattern[dim*loop + np.array(range(dim))]
	path[loop] = np.argmax(v)
print("path: ",path)
print("energy: ",-(np.matmul(np.matmul(pattern.T,W),pattern)/2 - np.sum(pattern)*threshold))