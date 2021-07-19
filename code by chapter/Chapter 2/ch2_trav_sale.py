#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:49:42 2021

@author: tom verguts
The travelling salesperson, with some help from Hopfield
"""

import numpy  as np
import tensorflow as tf

d12, d13,  d23 = 1, 4, 4
d = np.array([[0, d12, d13], [d12, 0, d23], [d13, d23, 0]]) # distances
g = 3 # importance of sum constraint

W = np.zeros((9, 9), dtype = np.double)
W[0, 0]   = 2*g
W[1, 0:2] = [g, 2*g]
W[2, 0:3] = [g, g, 2*g]
W[3, 0:4] = [g, d[0, 1], 0, 2*g]
W[4, 0:5] = [d[0, 1], g, d[0, 1], g, 2*g]
W[5, 0:6] = [0, d[0, 1], g, g, g, 2*g]
W[6, 0:7] = [g, d[0, 2], 0, g, d[1, 2], 0, 2*g]
W[7, 0:8] = [d[0, 2], g, d[0, 2], d[1, 2], g, d[1, 2], g, 2*g]
W[8, 0:9] = [0, d[0, 2], g, 0, d[1, 2], g, g, g, 2*g]
W = -W
W = W + W.T - np.diag(np.diag(W))

start_pattern = (np.random.random(9)>0.5)*1 # start in random state
start_pattern = start_pattern[:, np.newaxis]
print(start_pattern)

# a function to sample the network iteratively
def hopfield(start_pattern = None, n_sample = 0):
	pattern = tf.cast(start_pattern, dtype = tf.double)
	for loop in range(n_sample):
		net_input = tf.matmul(W, pattern) + tf.multiply(pattern, +4*g)
		clipped   = tf.cast(tf.math.greater(net_input, 0), tf.double)
		pattern   = clipped
	return pattern

pattern = hopfield(start_pattern = start_pattern, n_sample = 10)
pattern = np.array(pattern)
print(pattern)
path = np.ndarray(3)
for loop in range(3):
	v = pattern[3*loop + np.array(range(3))]
	path[loop] = np.argmax(v)
print("path: ",path)
print("energy: ",-(np.matmul(np.matmul(pattern.T,W),pattern) + np.sum(pattern)*4*g))