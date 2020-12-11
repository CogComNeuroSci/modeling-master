#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi
"""

import numpy as np
import time
from ch0_course_functions import plot_network, update_network

timesleep = .05
n_tsteps = 50
times = np.arange(n_tsteps)
t = 1
# output units
ydog = np.zeros(n_tsteps)
ycat = np.zeros(n_tsteps)
incat = np.zeros(n_tsteps)
indog = np.zeros(n_tsteps)
# std of noise
sigma = .5
# learning rate
alpha = .2

# we have 3 input unit on the first layer and 2 output units on the second layer
layers = np.array([1, 1, 1, 2, 2])
n_units = len(layers)
# here we set all the input activations (index from 0 to 2) to 0.
# We also set the two output units ('cat', index 3, and 'dog', index 4) to 0.
activations = np.zeros(n_units)
weights = np.zeros(shape=[n_units, n_units])

# setting up weights for the cat-dog detector
weights[3, 0] = .8 # cats often bite visitors
weights[4, 0] = .1 # dogs rarely bite visitors
weights[3, 1] = .2 # cats often have four legs
weights[4, 1] = .2 # dogs often have four legs
weights[3, 2] = .1 # cats rarely have their pictures on FB
weights[4, 2] = .8 # dogs often have their pictures on FB
weights[4, 3] = -.2 # a cat cannot be a dog, and vice versa


# let's add the inhibition between cat and dog
weights[4, 3] = -0.3 # dog inhibits cat
weights[3, 4] = -0.3 # cat inhibits dog

# let's input a certain pattern of activations (i.e. x1, x2 and x3)
activations[:3] = 1, 1, 0

# computing the initial y activation values
incat = weights[3, 0] * activations[0] + weights[3, 1] * activations[1] + weights[3, 2] * activations[2]
indog = weights[4, 0] * activations[0] + weights[4, 1] * activations[1] + weights[4, 2] * activations[2]
ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
activations[3:] = [ycat[t], ydog[t]]
energy = -incat*ycat[t] - indog*ydog[t] - weights[4, 3]*ycat[t]*ydog[t]

# plot the network
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights, layers = layers, energy = energy)


for t in times[1:]:
    ycat[t] = ycat[t-1] + alpha * (incat + weights[3, 4] * ydog[t-1]) + np.random.randn()*sigma
    ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
    if ycat[t]<0 : ycat[t] = 0
    if ydog[t]<0: ydog[t] = 0
    activations[3:] = [ycat[t], ydog[t]]
    energy = -incat*ycat[t] - indog*ydog[t] - weights[4, 3]*ycat[t]*ydog[t]
    
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations,
        unit_pos = unit_pos, weights = weights, layers = layers, change = 0,
        cycle = t, energy = energy, learn_trial_n = -1)

    time.sleep(timesleep)


