#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:47:06 2020

@author: tom verguts
TF2-based hopfield network, activation and learning equations (i.e., combining chapters 2 and 3)
"""

#%% import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%% initialize and define data and functions
dim = 1 # 1: 1-dim vectors; 2: 2-dim MNIST numbers
def define_length(pattern):
	length = 1
	for loop in pattern.shape[1:]:
		length *= loop
	return length	

# define data
if dim == 1:
	 # 1D vectors
    train_pattern = np.array([[1, 1, 1, -1, -1, -1], [-1, -1, -1, +1, +1, +1]])
    start_pattern = np.array([-1, 1,  1, -1, 1, -1])
    length = define_length(train_pattern)
else:
    # 2D numbers from MNIST data set
    all_data = tf.keras.datasets.mnist.load_data()
    start_number, stop_number = 14, 19
    n_numbers    = stop_number - start_number
    train_pattern = all_data[0][0][start_number:stop_number] # first n_numbers numbers from the training data
    train_pattern = 2*(train_pattern>5) - 1  # recode to -1 / +1 patterns
    length = define_length(train_pattern)
    start_pattern = np.array(np.random.choice([-1, 1], size = length))


# define variables
n_train   = 50 # how many training steps to take
n_samples_test = 4 # how often to step in activation space
x_pattern= np.zeros((n_samples_test, length))
w        = tf.Variable(np.random.randn(length, length).astype(np.float32)/10)
update_w = tf.Variable(np.random.randn(length, length).astype(np.float32)/10)
b        = tf.Variable(np.random.randn(1, length).astype(np.float32)/10)
update_b = tf.Variable(np.random.randn(1, length).astype(np.float32)/10)
weights  = np.ndarray((n_train//5, length, length))
bias     = np.ndarray((n_train//5, length))

# a function to sample the network iteratively
def hopfield_sample(start_pattern = None, n_sample = 10):
	pattern = start_pattern
	pattern= tf.convert_to_tensor(pattern, dtype = tf.float32)
	pattern = tf.reshape(pattern, [1, length])
	for loop in range(n_sample):
		net_input = tf.matmul(w, pattern, transpose_b=True) + tf.transpose(tf.multiply(pattern, b))
		clipped   = tf.multiply(2.,tf.cast(tf.math.greater(net_input, 0), tf.float32))-1
		pattern = tf.transpose(clipped)
	return tf.squeeze(pattern)

def hopfield_train(pattern = None):
	pattern= tf.convert_to_tensor(pattern, dtype = tf.float32)
	pattern = tf.reshape(pattern, [1, length])
	update_w.assign(tf.matmul(tf.transpose(pattern), pattern))
	update_b.assign(pattern)	
	return [update_w, update_b]

#%% preprocess
if dim > 1:# dimension of input pattern
	train_pattern =  train_pattern.reshape(train_pattern.shape[0], length)
	start_pattern = start_pattern.reshape(length)
	
# main routine: training	
for loop in range(n_train):
	nr = np.random.randint(train_pattern.shape[0]) # a random training pattern
	update_w, update_b = hopfield_train(pattern = train_pattern[nr])
	w.assign(w + update_w)
	b.assign(b + update_b)
	if not loop%(n_train//4): # check out the weight matrix
		indx = loop//(n_train//4)
		weights[indx] = w.numpy()
		bias[indx]   = b.numpy()

# training is over: sample novel data pattern novel_x		
novel_x = start_pattern	
for loop in range(n_samples_test):
	x_pattern[loop] = novel_x
	novel_x = hopfield_sample(novel_x)

# %% print results for 1D vectors, or plot results for MNIST data (i.e., if dim = 2)
if dim == 1:
    print("start pattern: {}\n".format(start_pattern))
    print("end pattern: {}\n".format(novel_x))

if dim == 2:
    # plot intermediate weight matrices
    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    fig.suptitle("weight matrices during training")
    for loop in range(n_train//(n_train//4)):
        row, col = loop//2, loop%2
        ax[row, col].imshow(weights[loop])

    # plot intermediate patterns, for the final weight matrix
    fig, ax = plt.subplots(nrows = 1, ncols = 4)
    fig.suptitle("what does a random pattern drift toward?")
    for loop in range(n_samples_test):
        x_image = x_pattern[loop].reshape(28, 28)
        ax[loop].set_xticks([])
        ax[loop].set_yticks([])
        ax[loop].set_title("sample {}".format(loop))
        ax[loop].imshow(x_image, cmap = "gray")
        ax[loop].set_axis_off() # it's not a function plot, so nicer w/o axes

    # plot training pattern
    nrows = int(np.floor(n_numbers/3) + (n_numbers%3 > 0))
    ncols = int(np.minimum(n_numbers, 3))
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols)
    fig.suptitle("training patterns")
    for loop in range(n_numbers):
        row, col = loop//3, loop%3
        x_image = train_pattern[loop].reshape(28, 28)
        try: 
            ax[row, col].imshow(x_image)
        except:
            ax[loop].imshow(x_image)	