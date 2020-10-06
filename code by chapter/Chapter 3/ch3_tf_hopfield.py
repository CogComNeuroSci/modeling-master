#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:47:06 2020

@author: tom verguts
TF-based hopfield network, activation and learning equations
"""

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

# define data
train_pattern = np.array([-1, -1, 1, -1, -1, -1])
start_pattern = np.array([-1, 1,  1, -1, 1, -1])
size = train_pattern.size

# define variables
n_samples = 10 # how often to step in activation space
w    = tf.Variable(np.random.randn(size, size).astype(np.float32)/1000)
b    = tf.Variable(np.random.randn(1, size).astype(np.float32)/1000)

# a function to sample the network iteratively
def hopfield_sampl(start_pattern = None, n_sample = 10):
	pattern = start_pattern
	for loop in range(n_sample):
		net_input = tf.matmul(w, pattern, transpose_b=True) + tf.transpose(tf.multiply(pattern, b))
		clipped   = tf.multiply(2.,tf.cast(tf.math.greater(net_input, 0), tf.float32))-1
		pattern = tf.transpose(clipped)
	return pattern

def hopfield_train(pattern = None):
	update_w  = tf.assign(w, w + tf.matmul(tf.transpose(pattern), pattern))
	update_b  = tf.assign(b, b + pattern)	
	return [update_w, update_b]

# computation graph definition
x             = tf.placeholder(tf.float32, shape=[1, size])
hopfield_sa   = hopfield_sampl(start_pattern = x, n_sample = n_samples)
hopfield_tr   = hopfield_train(pattern = x)
init          = tf.global_variables_initializer()

# preprocess
if len(train_pattern.shape)>1:
	train_pattern =  train_pattern.reshape(size)
	start_pattern  = start_pattern.reshape(size)
	
# main routine	
with tf.Session() as sess:
	sess.run(init)
	for loop in range(20):
		res = sess.run(hopfield_sa, feed_dict = {x: [start_pattern]})
		print(res)
		sess.run(hopfield_tr, feed_dict = {x: [train_pattern]})
	weights = w.eval()
		
# plot result
fig, ax = plt.subplots()
ax.imshow(weights)

		