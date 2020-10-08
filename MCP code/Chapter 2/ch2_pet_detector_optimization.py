#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi
"""

import numpy as np
import time
from ch0_course_functions import plot_network, update_network, plot_network2
    
timesleep = .1
n_tsteps = 50
times = np.arange(n_tsteps)
t = 1
# output units
ydog = np.zeros(n_tsteps)
ycat = np.zeros(n_tsteps)
incat = np.zeros(n_tsteps)
indog = np.zeros(n_tsteps)
# std of noise
sigma = .2
# scaling parameter
alpha = .2

# we have 3 input unit on the first layer and 2 output units on the second layer
layers = np.array([1, 1, 1, 2, 2])
n_units = len(layers)
# here we set the first feature detector ('bites visitors') to 1 and the 2
# others to 0. We also set the two output units ('cat' and 'dog') to 0 and we
# will update their activation later
activations = np.array([0., 1., 1., 0., 0.])
weights = np.zeros(shape=[n_units, n_units])

weights[3, 0] = .8 # cats often bite visitors
weights[4, 0] = .1 # dogs rarely bite visitors
weights[3, 1] = .2 # cats often have four legs
weights[4, 1] = .2 # dogs often have four legs
weights[3, 2] = .1 # cats rarely have their pictures on FB
weights[4, 2] = .8 # dogs often have their pictures on FB
weights[4, 3] = -.2 # a cat cannot be a dog, and vice versa

# computing the initial y activation values
# eq. 2.1
incat = weights[3, 0] * activations[0] + weights[3, 1] * activations[1] + weights[3, 2] * activations[2]
# eq. 2.2
indog = weights[4, 0] * activations[0] + weights[4, 3] * activations[1] + weights[4, 2] * activations[2]
# eq. 2.4
ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
# eq. 2.5
ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
# this is just for plotting: put the ydog and ycat values at their respective 
# index in the activation array to plot these values on the network figure
activations[3:] = [ycat[t], ydog[t]]
# eq. 2.3
energy = -incat*ycat[t] - indog*ydog[t] - weights[4, 3]*ycat[t]*ydog[t]

# plot the network
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network2(figsize = [13, 7], activations = activations,
                  weights = weights, layers = layers, energy = energy)

#%%
for t in times[1:]:
    # update the ycat value (eq. 2.4)
    ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
    # update the ydog value (eq. 2.5)
    ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
    # rectify the y values: if they are smaller than zero put them at zero
    if ycat[t]<0 : ycat[t] = 0
    if ydog[t]<0: ydog[t] = 0
    # this is just for plotting: put the new ydog and ycat at their respective 
    # index in the activation array to plot the updated values on the network
    # plot
    activations[3:] = [ycat[t], ydog[t]]
    # compute the new energy of the network (eq. 2.3)
    energy = -incat*ycat[t] - indog*ydog[t] - weights[4, 3]*ycat[t]*ydog[t]
    
    # update the figure of the network
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations,
        unit_pos = unit_pos, weights = weights, layers = layers, change = 0,
        cycle = t, energy = energy)
    
    # wait a bit to be able to see the changes
    time.sleep(timesleep)


