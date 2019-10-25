#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi

To implement hebbian learning with  matrix notation (more specifically dot product)
we can either use the same format as in the ch3_pet_detector_hebbian_and_optimization_solution.py script
in which the samples were of size 3-by-1 and the targets were of size 2-by-1.
If we do a dot product between these matrices we will get a 3-by-2 matrix of
of delta weights (the change that we need to make to the current weight matrix)
This would work fine and it fits with the way it is described in chapter 3.
But then we would need to change the format of this matrix to be able to plot
the network with the plot_network and update_network functions.

So another solution, if we want to plot the network more easily, is to work with
a weight matrix of n_units-by-n_units (5-by-5 in our pet detector) therefore
another way of doing the matrix notation is to consider that each sample is
5-by-5 matrix and each target is also a 5-by-5 matrix.
This is the solution we will use in this script in order to easily plot the network.

Again the first solution and the one  provided here are equivalent in the
outcome of the network. The first one is more parsimonious and fits witht the
chapter's descriptions and the 2nd one allows us to easily plot the network.
"""

import numpy as np
import time
from matplotlib import pyplot as pl
from ch0_course_functions import plot_network, update_network


timesleep = .1

# we have 3 input unit on the first layer and 2 output units on the second layer
layers = np.array([1, 1, 1, 2, 2])
n_units = len(layers)
# here we set all the input activations (index from 0, 1 and 2) to 0.
# We also set the two output units ('cat', index 3, and 'dog', index 4) to 0.
activations = np.array([0., 0., 0., 0., 0.])

# let's set energy to zero for now
energy = 0


###############################################################################
####    LEARNING PART
###############################################################################

# our learning parameter
beta = .8

# learning samples (activation (x) of each input unit)
train_samples = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])

# the targets (basically representing "dog" or "cat"):
    #                      cat                   cat                cat        
targets = np.array(  [[0, 0, 0, 1, 0],   [0, 0, 0, 1, 0],    [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],    [0, 0, 0, 0, 1],    [0, 0, 0, 0, 1]])
#                         dog                    dog                dog


# how many learning samples do we have (hence, how many trials are we going to do?)
n_trials = train_samples.shape[0]
# get the number of dimensions (i.e. units) in the samples
n_sample_dim = train_samples.shape[1]
# get the number of dimensions (i.e. units) in the targets
n_target_dim = targets.shape[1]

# create the weight matrix of size n_units-by-n_units
weights = np.zeros(shape = [n_units, n_units])

# let's set a SMALL weight everywhere so that the plotting functions have
# something other than zeros otherwise we won't see the weights in the upper-right plot
weights += 0.01

# plot the network to see how it initially looks like
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights, layers = layers, energy = 0)

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    # the dot product!
    # here we have to use "np.newaxis" because when we select one
    # train_sample (train_samples[trial_n, :]) instead of having a 5-by-1 array
    # python transforms it into an array of size (5,) (or 5-by-0)
    # (meaning that there is only one dimension, probably because it is easier
    # to store or takes less space). But for our dot product we need it to
    # be 5-by-1 so we use np.newaxis to say "I know there is nothing there but
    # consider it as a dimension anyway". Therefore the size of 
    # train_samples[trial_n, :][:, np.newaxis] is 5-by-1 (or (5, 1) in python
    # notation if you do train_samples[trial_n, :][:, np.newaxis].shape)
    # the same thing applies for target[trial_n, :]
    weights = weights + beta * np.dot(targets[trial_n, :][:, np.newaxis],
                                              train_samples[trial_n, :][:, np.newaxis].T)
    
    ## Update the network plot
    # Here we fixed the parameter "cycle" to 0 because we use it to represent
    # the activation optimization cycle as seen in chapter 2 (which is not what
    # we are doing in this loop).
    # Also we added a parameter "learn_trial_n" which represents the cycle/trial
    # of learning we are in, in other words the learning trial we are using.
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations, change =0,
        unit_pos = unit_pos, weights = weights, layers = layers,
        cycle = 0, learn_trial_n = trial_n+1, energy = energy)

    # wait for a short time to see the changes
    time.sleep(timesleep)
    
# we add a negative weight between ydog and ycat by hand because the hebbian 
# learning algorithm cannot create it
weights[4, 3] = -.2

# update the network plot
update_network(fig = fig, axs = axs, texts_handles = texts_handles,
    lines_handles = lines_handles, activations = activations, change =0,
    unit_pos = unit_pos, weights = weights, layers = layers,
    cycle = 0, learn_trial_n = trial_n+1, energy = energy)


pl.suptitle('Learning phase finished!')
fig.canvas.draw()
time.sleep(.5)

#%%############################################################################
####    TESTING PART
###############################################################################

pl.suptitle('Let\'s now see how our trained network behaves!')
fig.canvas.draw()


n_tsteps = 20
times = np.arange(n_tsteps)
t = 1

# output units
ydog = np.zeros(n_tsteps)
ycat = np.zeros(n_tsteps)

# std of noise
sigma = .7
# scaling parameter
alpha = .2


# let's input a certain pattern of activations (i.e. x1, x2 and x3)
activations[:3] = 1, 1, 0

################################
################################
#### USING THE DOT PRODUCT
################################
################################
# computing the initial y activation values
# here we only keep the last 2 values of the dot product result because otherwise
# we will also update the input values and we do not want this (see last slide of lesson 4 presentation)
activations[3:] = np.dot(weights, activations)[3:]
################################
################################

# eq. 2.1
incat = weights[3, 0] * activations[0] + weights[3, 1] * activations[1] + weights[3, 2] * activations[2]
# eq. 2.2
indog = weights[4, 0] * activations[0] + weights[4, 1] * activations[1] + weights[4, 2] * activations[2]
# eq. 2.4
ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
# eq. 2.5
ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
# this is just for plotting: put the ydog and ycat values at their respective 
# index in the activation array to plot these values on the network figure
activations[3:] = [ycat[t], ydog[t]]
# eq. 2.3
energy = -incat*ycat[t] - indog*ydog[t] - weights[4, 3]*ycat[t]*ydog[t]

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
    
    ## Update the network plot
    # Here we update the parameter "cycle" on every iteration because we use it
    # to represent the activation optimization cycle as seen in chapter 2.
    # The parameter "learn_trial_n" which represents the cycle/trial of
    # learning we are in, is fixed because learning has been done already.
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations,
        unit_pos = unit_pos, weights = weights, layers = layers, change = 0,
        cycle = t, energy = energy, learn_trial_n = trial_n+1)

    time.sleep(timesleep)


