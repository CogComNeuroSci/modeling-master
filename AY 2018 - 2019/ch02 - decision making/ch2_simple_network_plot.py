#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:36:15 2018

@author: mehdisenoussi
"""

import numpy as np
import time
from ch0_course_functions import plot_network

# we want 2 input units and 1 output units so we create
#   - an array "layers" of length 3, to store which unit belongs to which layer
#   - an array "activations" of length 3, to store the activation of each unit
layers = np.array([1, 1, 2])

n_units = 3
activations = np.zeros(shape = (n_units))

# let's put input unit 1 at 1 and input unit 2 at 0
activations[0] = 1
activations[1] = 0

# let's now create a 2-D matrix for the weights
weights = np.zeros(shape = (n_units, n_units))
# let's put weight from unit 1 to output unit at .3
weights[0, 2] = .3
# let's put weight from unit 2 to output unit at .8
weights[1, 2] = .8

# create the activation function of the output unit
weight_output = np.dot(weights.T, activations)
activations[2] = weight_output[-1]

# plot the network using the function plot_network
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = None, activations = activations,
                  weights = weights, layers = layers)


