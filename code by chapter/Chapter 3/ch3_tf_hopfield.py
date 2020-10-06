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
train_pattern = np.array([[1, 1, 1, -1, -1, -1], [-1, -1, -1, +1, +1, +1]])
start_pattern = np.array([-1, 1,  1, -1, 1, -1])
dim = 1
length = 1
for loop in range(length):
	length *= train_pattern.shape[loop+1]

# define variables
n_train   = 20 # how many training steps to take
n_samples = 5 # how often to step in activation space
w    = tf.Variable(np.random.randn(length, length).astype(np.float32)/1)
b    = tf.Variable(np.random.randn(1, length).astype(np.float32)/1)
weights = np.ndarray((n_train//5, length, length))

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
x             = tf.placeholder(tf.float32, shape=[1, length])
hopfield_sa   = hopfield_sampl(start_pattern = x, n_sample = n_samples)
hopfield_tr   = hopfield_train(pattern = x)
init          = tf.global_variables_initializer()

# preprocess
if dim > 1:# dimension of input pattern
	train_pattern =  train_pattern.reshape(train_pattern.size[0], length)
	start_pattern  = start_pattern.reshape(train_pattern.size[0], length)
	
# main routine	
with tf.Session() as sess:
	sess.run(init)
	for loop in range(n_train):
		nr = np.random.randint(train_pattern.shape[0]) # a random training pattern
		sess.run(hopfield_tr, feed_dict = {x: [train_pattern[nr]]})
		if not loop%5: # check out the weight matrix
			res = sess.run(hopfield_sa, feed_dict = {x: [start_pattern]})
			print(res)
			indx = loop//5
			weights[indx] = w.eval()
		
# plot intermediate weight matrices
fig, ax = plt.subplots(nrows = 2, ncols = 2)
for loop in range(n_train//5):
	row = loop//2
	col = loop%2
	ax[row, col].imshow(weights[loop])

		