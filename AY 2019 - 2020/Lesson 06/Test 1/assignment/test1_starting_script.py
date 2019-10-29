#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi

Write your answers to the test questions here.
...

"""

# this script is divided in two parts
    # 1) a learning part in which we use hebbian learning to train the model, 
    #    i.e change its weights
    #
    # 2) a testing part in which we present cat and dog prototypes (arrays of 
    #    input units) to this trained model and optimize the output activations

import numpy as np

# how many units does this network have in total?
n_units = 5

#%%
###############################################################################
####    1) LEARNING PART
###############################################################################

# our learning parameter is set at 0.1
beta = 0.1

# Because we will use the dot product to compute the delta weights we will use
# matrices that represent the activations of all units in the network (both 
# inputs and output units). See figure below question 1 of the test instructions.

# Define the cat prototype
cat_prototype = np.array([1, 0.5, 0, 0, 0])
# Define the dog prototype
dog_prototype = np.array([0, 0.5, 1, 0, 0])
# set the number of samples we want to present of the cat prototype
n_cat_samples = 10
# set the number of samples we want to present of the dog prototype
n_dog_samples = 10

train_cat_samples = np.tile(cat_prototype, (n_cat_samples, 1))
train_dog_samples = np.tile(dog_prototype, (n_cat_samples, 1))
train_samples     = np.vstack((train_cat_samples, train_dog_samples))

## Add random normally distributed noise (mean = 0, standard deviation = 0.01)
# To do this we create an array of size (n_cat_samples + n_dog_samples)-by-3 (
# because we have 3 input units) to add some random noise to each value in the
# train_sample array.
# The function np.random.randn() returns a matrix of random normally distributed
# values of mean = 0 and standard deviation = 1.
# We then multiply these values by the standard deviation we want in our
# training samples.
samples_noise_std = 0.01
train_samples[:, :3] = train_samples[:, :3] + np.random.randn(n_cat_samples + n_dog_samples, 3) * samples_noise_std

# the targets (basically representing "dog" or "cat"):
# Define the cat target
cat_target = np.array([0, 0, 0, 1, 0])
# Define the dog target
dog_target = np.array([0, 0, 0, 0, 1])
# create all the targets
train_cat_targets = np.tile(cat_target, (n_cat_samples, 1))
train_dog_targets = np.tile(dog_target, (n_cat_samples, 1))
targets = np.vstack((train_cat_targets, train_dog_targets))


# create a weight matrix of size n_units-by-n_units
weights = np.zeros(shape = [n_units, n_units])

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    weights = weights + beta * np.dot(targets[trial_n, :][:, np.newaxis],
                                              train_samples[trial_n, :][:, np.newaxis].T)

# we add a negative weight between ydog and ycat "by hand"
weights[4, 3] = -0.2


#%%
###############################################################################
####    2) TESTING PART
###############################################################################

# the fixed number of time steps for which we will optimize the output unit activations
n_tsteps = 100
# create an array with all these time steps
times = np.arange(n_tsteps)
# startig index for the time steps since for the first optimizationn step we
# need to use the previous time step's activations
t = 1

# standard deviation of random normally distributed noise
sigma = 0.5
# update rate
alpha = 0.2

# how many test inputs will we use? (n_test/2 for each prototype)
n_test_total = 20
# use n_test/2 cat_prototype and n_test/2 dog_prototypes as inputs
test_inputs = np.vstack((np.tile(cat_prototype, (int(n_test_total/2), 1)),
                         np.tile(dog_prototype, (int(n_test_total/2), 1))))

# we add some noise so that they are all a bit different
test_inputs[:, :3] = test_inputs[:, :3] + np.random.randn(n_test_total, 3) * samples_noise_std


# array to store the results of the activation optimization
output_result = np.zeros(shape = (n_test_total, 2))

# loop over all the test inputs
for test_input_n in np.arange(n_test_total):
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
    
    # store the results of the network optimization
    output_result[test_input_n, :] = [ycat[-1], ydog[-1]]



# the np.round function is used to simplify the output, it limits the number
# after the floating point to only one number
print(np.round(output_result, 1))




