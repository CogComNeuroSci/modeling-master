#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July 19, 2021

@author: tom verguts
TF-based hopfield network for arbitrary vectors X1, X2, ...
Both X_i and -X_i are attractors in this case.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define fixed parameters
n_units   = 100
n_stimuli = 3
threshold = 0
learning_rate = 1

# define the stimuli
X = np.ndarray((n_units, n_stimuli))
for stimulus_loop in range(n_stimuli):
	X[:,stimulus_loop] = 2*(np.random.random(n_units)>0.5)-1 # +1 / -1 coding

# construct weight matrix
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

# a function to sample the network iteratively
def hopfield(start_pattern = None, n_sample = 0):
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
print("distance to different stimuli (0-1 scale): ", distance[-1,:])
for stim_loop in range(n_stimuli):
	  plt.plot(distance[:, stim_loop], color = "black")