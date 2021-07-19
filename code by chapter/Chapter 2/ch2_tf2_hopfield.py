#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:47:06 2020

@author: tom verguts
TF-based hopfield network for the john - male - poor - mary - female - rich example
"""

import tensorflow as tf
import numpy as np

# define fixed parameters
inh  = -0.5
row1 = np.array([1, 1, 1, inh, inh, inh])
row2 = np.array([inh, inh, inh, 1, 1, 1])
w    = np.array((row1, row1, row1, row2, row2, row2)).astype(np.float32)
threshold = -1
b    = np.array(row1.size*[threshold])
b    = b[:, np.newaxis]
start_pattern = np.array([1, 1, 1, 1, 0, 0])
start_pattern = start_pattern[:, np.newaxis]

# a function to sample the network iteratively
def hopfield(start_pattern = None, n_sample = 0):
	pattern = tf.cast(start_pattern, dtype = tf.float32)
	for loop in range(n_sample):
		net_input = tf.matmul(w, pattern) + tf.multiply(pattern, b)
		clipped   = tf.cast(tf.math.greater(net_input, 0), tf.float32)
		pattern   = clipped
		print(pattern)
	return pattern

pattern = hopfield(start_pattern = start_pattern, n_sample = 10)
print(pattern.numpy())