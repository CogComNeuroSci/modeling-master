#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July 19, 2021

@author: tom verguts
TF-based hopfield network for arbitrary vectors X1, X2, ...
Both X_i and -X_i are attractors in this case; see MCP book for an explanation why (Exercise 3.11)
"""

#%% import and initialisation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define fixed parameters
n_units   = 40
n_stimuli = 2
threshold = 0
learning_rate = 1

# define the stimuli
X = np.ndarray((n_units, n_stimuli))
for stimulus_loop in range(n_stimuli):
	X[:,stimulus_loop] = 2*(np.random.random(n_units)>0.5)-1 # +1 / -1 coding

# construct weight matrix via hebb-like learning rule
w = np.zeros((n_units, n_units))
for stimulus_loop in range(n_stimuli):
	w    = w + learning_rate*np.matmul(X[:, stimulus_loop][:,np.newaxis], X[:, stimulus_loop][:,np.newaxis].T)
b    = np.array(n_units*[threshold])
b    = b[:, np.newaxis]

# initial starting point
start_pattern = (2*np.random.random(n_units)>0.5)-1
start_pattern = start_pattern[:, np.newaxis]

n_sample = 50
distance = np.zeros((n_sample, n_stimuli))
max_distance = np.sqrt(4*n_units)

# %% main program

def hopfield(start_pattern = None, n_sample = 0):
	""""a function to sample the network iteratively"""
	pattern = tf.cast(start_pattern, dtype = tf.double)
	for loop in range(n_sample):
		for stimulus_loop in range(n_stimuli):
			distance[loop, stimulus_loop] = np.linalg.norm(X[:, stimulus_loop][:,np.newaxis]-pattern)/max_distance
		net_input = tf.matmul(w, pattern)  + tf.multiply(pattern, b)
		clipped   = tf.cast(tf.math.greater(net_input, 0), tf.float32)
		pattern   = clipped*2 - 1
		pattern = tf.cast(pattern, dtype = tf.double)
	return pattern

pattern = hopfield(start_pattern = start_pattern, n_sample = n_sample)

#%% print and plot results
print("distance to different stimuli (0-1 scale): ", distance[-1,:])
fig, ax = plt.subplots(nrows = 2, ncols = 1)

# X axis is time; Y-axis is distance to each of the stored stimuli. If an attractor stimulus is reached,
# the distance to that attractor should become zero (or very small)
ax[0].set_title("distance to stored stimuli across time") 
for stim_loop in range(n_stimuli):
	  ax[0].plot(distance[:, stim_loop], color = "black")
ax[0].set_xlabel("time")
ax[0].set_ylabel("distance")

# row 1 is the initial, random pattern; rows 2, 3 are the stored patterns; row 4 is the stimulus reached at the final time step
ax[1].set_title("start pattern (row 0), stored patterns, final pattern (row -1)")  
data_to_plot = np.column_stack((start_pattern, X, pattern.numpy()))
ax[1].imshow(data_to_plot.T)