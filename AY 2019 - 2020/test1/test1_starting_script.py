#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi

To write this script I used the ch3_pet_detector_hebbian_and_optimization_solution.py to start with.
In this script we will not use the matrix notation to implement hebbian learning
"""

# this script is divided in two parts
    # 1) a learning part in which we use hebbian learning to train the model, 
    #    i.e change its weights
    #
    # 2) a testing part in which we present cat and dog examplars (vectors of 
    #    input units) to this trained model and use the optimize the energy function.

import numpy as np
import time
from matplotlib import pyplot as pl

timesleep = .1

# how many units does this network have in total?
n_units = 5

###############################################################################
####    LEARNING PART
###############################################################################

# our learning parameter is set at 0.1
beta = .1

# learning samples (activation (x) of each input unit)
# 10 cats (i.e. [1, .5, 0]) and 10 dogs (i.e. [0, .5, 1])

# Define the cat prototype
cat_prototype = np.array([1, .5, 0, 0, 0])
# set the number of samples we want to present of this prototype
n_cat_samples = 10
# Define the dog prototype
dog_prototype = np.array([0, .5, 1, 0, 0])
# set the number of samples we want to present of this prototype
n_dog_samples = 10

# the function tile repeats a certain set of values in the same order a certain
# number of times but it creates a 1-dimensional array of these values so we use
# the method reshape() to rearrange them into an array of n_samples-by-n_values
# finally we use np.vstack() to put the cat and dog samples into the same array
train_samples = np.vstack([np.tile(cat_prototype, n_cat_samples).reshape(n_cat_samples, n_units),
                           np.tile(dog_prototype, n_dog_samples).reshape(n_dog_samples, n_units)])

# add gaussian noise with mean = 0 and standard deviation = 0.01
# we create an array of size (n_cat_samples + n_dog_samples)-by-3 to add some
# independent gaussian noise to each value in the train_sample array
# we also multiply these values by the standard deviation we want in our samples
samples_noise_std = 0.01
train_samples[:, :3] = train_samples[:, :3] + np.random.randn(n_cat_samples + n_dog_samples, 3) * samples_noise_std

# the targets (basically representing "dog" or "cat"):
# Define the cat target
cat_target = np.array([0, 0, 0, 1, 0])
# Define the dog target
dog_target = np.array([0, 0, 0, 0, 1])
# create the all the targets
targets = np.vstack([np.tile(cat_target, n_cat_samples).reshape([n_cat_samples, n_units]),
                     np.tile(dog_target, n_dog_samples).reshape([n_dog_samples, n_units])])


# how many learning samples do we have (hence, how many trials are we going to do?)
n_trials = train_samples.shape[0]
# get the number of dimensions (i.e. units) in the samples
n_sample_dim = train_samples.shape[1]
# get the number of dimensions (i.e. units) in the targets
n_target_dim = targets.shape[1]

# create a weight matrix of size n_units-by-n_units
weights = np.zeros(shape = [n_units, n_units])

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    
    weights = weights + beta * np.dot(targets[trial_n, :][:, np.newaxis],
                                              train_samples[trial_n, :][:, np.newaxis].T)

# we add a negative weight between ydog and ycat by hand
weights[4, 3] = -.2


#%%
###############################################################################
####    TESTING PART
###############################################################################

# the fixed number of time steps for which we will optimize the output unit activations
n_tsteps = 100
# create an array with all these time steps
times = np.arange(n_tsteps)
# startig index for the time steps since for the first optimizationn step we
# need to use the previous time step's activations
t = 1

# std of noise
sigma = .7
# learning rate
alpha = .2

# how many test inputs will we use? (n_test/2 for each prototype)
n_test = 10
# use n_test/2 cat_prototype and n_test/2 dog_prototypes as inputs
test_inputs = np.vstack([np.tile(cat_prototype, int(n_test/2)).reshape(int(n_test/2), n_units),
                           np.tile(dog_prototype, int(n_test/2)).reshape(int(n_test/2), n_units)])

# we add some noise so that they are all a bit different
test_inputs[:, :3] = test_inputs[:, :3] + np.random.randn(n_test, 3) * samples_noise_std


# array to store the results of the activation optimization
output_result = np.zeros(shape = (n_test, 2))

for test_input_n in np.arange(n_test):
    ydog = np.zeros(n_tsteps)
    ycat = np.zeros(n_tsteps)
    
    this_input = test_inputs[test_input_n, :]
    
    incat = weights[3, 0] * this_input[0] + weights[3, 1] * this_input[1] + weights[3, 2] * this_input[2]
    indog = weights[4, 0] * this_input[0] + weights[4, 1] * this_input[1] + weights[4, 2] * this_input[2]

    # update the ycat value (eq. 2.4)
    ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
    # update the ydog value (eq. 2.5)
    ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma

    # optimize the activation of the output units
    for t in times[1:]:
        ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
        ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
        # rectify the y values: if they are smaller than zero put them at zero
        if ycat[t]<0: ycat[t] = 0
        if ydog[t]<0: ydog[t] = 0
        
    # compute the new energy of the network (eq. 2.3)
    energy = -incat*ycat[t] - indog*ydog[t] - weights[4, 3]*ycat[t]*ydog[t]
    
    # store the results of the network optimization
    output_result[test_input_n, :] = [ycat[-1], ydog[-1]]



# the np.round function is used to simplify the output, it limits the number
# after the floating point to only one number
print(np.round(output_result, 1))




