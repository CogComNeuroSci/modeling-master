#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi
"""

import numpy as np
import time
from ch0_course_functions import plot_network, update_network
import matplotlib as pl

# we have 3 input unit on the first layer and 2 output units on the second layer
layers = np.array([1, 1, 1, 2, 2])
n_units = len(layers)
# here we set all the input activations (index from 0 to 2) to 0.
# We also set the two output units ('cat', index 3, and 'dog', index 4) to 0.
activations = np.array([0., 0., 0., 0., 0.])

# let's set energy to zero for now
energy = 0



###############################################################################
####    LEARNING PART
###############################################################################

# our learning scaling parameter
beta = .8

# training samples (activation (x) of each input unit)
train_samples = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])

# the targets (basically representing "dog" or "cat"):
    #                      cat                   cat                cat        
targets = np.array(  [[0, 0, 0, 1, 0],   [0, 0, 0, 1, 0],    [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],    [0, 0, 0, 0, 1],    [0, 0, 0, 0, 1]])
#                         dog                    dog                dog


# how many training samples do we have (hence, how many trials are we going to do?)
n_trials = train_samples.shape[0]
# get the number of dimensions (i.e. units) in the samples
n_sample_dim = train_samples.shape[1]
# get the number of dimensions (i.e. units) in the targets
n_target_dim = targets.shape[1]

# create the weight matrix (n_trials+1 because it has to be initialized)
weights = np.zeros(shape = [n_trials + 1, n_units, n_units])


# let's set random SMALL weights so that the plotting functions have something
# other than zeros
# random.random() yields a number between 0 and 1, then we divide by 10 so we
# get a number between 0 and 0.1
random_weight = np.random.random()/10.
weights[0, 3, 0] = random_weight # random weight 
weights[0, 4, 0] = random_weight
weights[0, 3, 1] = random_weight
weights[0, 4, 1] = random_weight
weights[0, 3, 2] = random_weight
weights[0, 4, 2] = random_weight

# create the matrix of zeros to erase the weights we don't want
dead_weights = np.logical_not(weights[0,:,:].astype(np.bool))




# plot the network to see how it initially looks like
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights[0, :, :], layers = layers, energy = 0)

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    weights[trial_n+1, :] = weights[trial_n, :] + \
                                beta * np.dot(targets[trial_n, :][:, np.newaxis], train_samples[trial_n, :][:, np.newaxis].T)
    
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations, change =0,
        unit_pos = unit_pos, weights = weights[trial_n+1, :, :], layers = layers,
        cycle = 0, learn_trial_n = trial_n+1, energy = energy)

    # to wait for any button press to go to the next iteration of the loop
    # you can make this "automatic" by changing the 0 to a number of seconds
    fig.waitforbuttonpress(0)



pl.suptitle('Learning phase finished!\nPress a key to input a certain pattern in the model and see how it behaves!')
fig.canvas.draw()
fig.waitforbuttonpress(0)

###############################################################################
####    TESTING PART
###############################################################################

# let's only keep the end result of our learning process
weights_end = weights[-1, :, :]


timesleep = .1
n_tsteps = 20
times = np.arange(n_tsteps)
t = 1

# output units
ydog = np.zeros(n_tsteps)
ycat = np.zeros(n_tsteps)
incat = np.zeros(n_tsteps)
indog = np.zeros(n_tsteps)

# std of noise
sigma = .7
# learning rate
alpha = .2

# let's add the inhibition between cat and dog
weights[4, 3] = -0.3 # dog inhibits cat
weights[3, 4] = -0.3 # cat inhibits dog

# let's input a certain pattern of activations (i.e. x1, x2 and x3)
activations[:3] = 1, 1, 0

################################
################################
#### USING THE DOT PRODUCT
################################
################################
# computing the initial y activation values
activations = np.dot(weights_end, activations)
################################
################################


incat = activations[3]
indog = activations[4]

ycat[t] = ycat[t-1] + alpha * (incat + weights_end[4, 3] * ydog[t-1]) + np.random.randn()*sigma
ydog[t] = ydog[t-1] + alpha * (indog + weights_end[4, 3] * ycat[t-1]) + np.random.randn()*sigma
activations[3:] = [ycat[t], ydog[t]]
energy = -incat*ycat[t] - indog*ydog[t] - weights_end[4, 3]*ycat[t]*ydog[t]

for t in times[1:]:
    ycat[t] = ycat[t-1] + alpha * (incat + weights_end[3, 4] * ydog[t-1]) + np.random.randn()*sigma
    ydog[t] = ydog[t-1] + alpha * (indog + weights_end[4, 3] * ycat[t-1]) + np.random.randn()*sigma
    if ycat[t]<0 : ycat[t] = 0
    if ydog[t]<0: ydog[t] = 0
    activations[3:] = [ycat[t], ydog[t]]
    energy = -incat*ycat[t] - indog*ydog[t] - weights_end[4, 3]*ycat[t]*ydog[t]
    
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations,
        unit_pos = unit_pos, weights = weights_end, layers = layers, change = 0,
        cycle = t, energy = energy, learn_trial_n = -1)

    time.sleep(timesleep)


