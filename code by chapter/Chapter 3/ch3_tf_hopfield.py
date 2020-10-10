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


def define_length(pattern):
	length = 1
	for loop in pattern.shape[1:]:
		length *= loop
	return length	

# define data: simple 1D pattern
# dim = 1
# train_pattern = np.array([[1, 1, 1, -1, -1, -1], [-1, -1, -1, +1, +1, +1]])
# start_pattern = np.array([-1, 1,  1, -1, 1, -1])
# length = define_length(train_pattern)

# define data: numbers from MNIST data set
dim = 2
all_data = tf.keras.datasets.mnist.load_data()
start_number = 11
stop_number  = 12
n_numbers    = stop_number - start_number
train_pattern = all_data[0][0][start_number:stop_number] # first n_numbers numbers
length = define_length(train_pattern)
start_pattern = np.array(np.random.choice([-1, 1], size = length))
x_pattern = np.ndarray((9, length))

# define variables
n_train   = 100 # how many training steps to take
n_samples = 1 # how often to step in activation space
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
	return tf.squeeze(pattern)

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
	train_pattern =  train_pattern.reshape(train_pattern.shape[0], length)
	start_pattern  = start_pattern.reshape(length)
	
# main routine	
with tf.Session() as sess:
	sess.run(init)
	for loop in range(n_train):
		nr = np.random.randint(train_pattern.shape[0]) # a random training pattern
		sess.run(hopfield_tr, feed_dict = {x: [train_pattern[nr]]})
		if not loop%(n_train//4): # check out the weight matrix
			indx = loop//(n_train//4)
			weights[indx] = w.eval()
	# training is over		
	novel_x = start_pattern	
	for loop in range(9):
		x_pattern[loop] = novel_x
		novel_x = sess.run(hopfield_sa, feed_dict = {x: [novel_x]})

		
# plot intermediate weight matrices
fig, ax = plt.subplots(nrows = 2, ncols = 2)
for loop in range(n_train//(n_train//4)):
	row = loop//2
	col = loop%2
	ax[row, col].imshow(weights[loop])

# plot intermediate patterns, for the final weight matrix
fig, ax = plt.subplots(nrows = 3, ncols = 3)
fig.suptitle("try to store {} digits with hopfield".format(n_numbers))
for loop in range(9):
	row = loop//3
	col = loop%3
	x_image = x_pattern[loop].reshape(28, 28)
	ax[row, col].imshow(x_image)
		