#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:47:06 2020

@author: tom verguts
TF-based hopfield network for the john - male - poor - mary - female - rich example
"""

import tensorflow.compat.v1 as tf
import numpy as np

# define fixed parameters
inh = -0.5
row1 = np.array([1, 1, 1, inh, inh, inh])
row2 = np.array([inh, inh, inh, 1, 1, 1])
w    = np.array((row1, row1, row1, row2, row2, row2)).astype(np.float32)
threshold = -1
b    = np.array(row1.size*[threshold])
start_pattern = [1, 1, 0, 0, 0, 1]

# a function to sample the network iteratively
def hopfield(start_pattern = None, n_sample = 10):
	pattern = start_pattern
	for loop in range(n_sample):
		net_input = tf.matmul(w, pattern, transpose_b=True) + tf.transpose(tf.multiply(pattern, b))
		clipped   = tf.cast(tf.math.greater(net_input, 0), tf.float32)
		pattern = tf.transpose(clipped)
	return pattern

# computation graph definition
x             = tf.placeholder(tf.float32, shape=[1, 6])
hopfield_rep  = hopfield(start_pattern = x, n_sample = 5)
init          = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	res = sess.run(hopfield_rep, feed_dict = {x: [start_pattern]})
	print(res)