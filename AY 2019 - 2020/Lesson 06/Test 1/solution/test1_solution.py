#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi

Write your answers to the test questions here.

1)
    The model reaches a correct pet detection 9/10 times. The test inputs for
    which it does not converge to the correct pet are due to noise.
    
    The model reaches a correct pet detection 6/10 times. Its performance is
    lower than with sigma = 0.7. This is due to the fact that when there is
    more noise in the network the incorrect output unit can by chance reach a
    higher activation and therefore inhibit the activation of the correct
    output unit.
    
4)
    The model reaches a correct pet detection 10/10 times. The model yields a
    better performance for pet detection with 5 than with 3 input units.
    This is because it has more input units, and therefore more information to
    detect which pet it is.
    A second reason why it performs better is that it has been trained on more
    samples (10 by prototype in question 1 and 20 in question 4) which
    increases the weights and therefore leads to better performances.

"""

# this script is divided in two parts
    # 1) a learning part in which we use hebbian learning to train the model, 
    #    i.e change its weights
    #
    # 2) a testing part in which we present cat and dog examplars (arrays of 
    #    input units) to this trained model and optimize the energy function.

import numpy as np

# how many units does this network have in total?
n_units = 7

#%%
###############################################################################
####    1) LEARNING PART
###############################################################################

# our learning parameter is set at 0.1
beta = 0.1

# Because we will use the dot product to compute the delta weights we will use
# matrices that represent the activations of all units in the network (both 
# inputs and output units)

# Define the cat prototype
cat_prototype = np.array([1, 0.5, 0, 0.8, 0.1, 0, 0])
# set the number of samples we want to present of this prototype
n_cat_samples = 20
# Define the dog prototype
dog_prototype = np.array([0, 0.5, 1, 0.1, 0.8, 0, 0])
# set the number of samples we want to present of this prototype
n_dog_samples = 20

train_cat_samples = np.tile(cat_prototype, (n_cat_samples, 1))
train_dog_samples = np.tile(dog_prototype, (n_cat_samples, 1))
train_samples     = np.vstack((train_cat_samples, train_dog_samples))

# add gaussian noise with mean = 0 and standard deviation = 0.01
# we create an array of size (n_cat_samples + n_dog_samples)-by-3 to add some
# independent gaussian noise to each value in the train_sample array
# we also multiply these values by the standard deviation we want in our samples
samples_noise_std = 0.01
train_samples[:, :5] = train_samples[:, :5] + np.random.randn(n_cat_samples + n_dog_samples, 5) * samples_noise_std

# the targets (basically representing "dog" or "cat"):
# Define the cat target
cat_target = np.array([0, 0, 0, 0, 0, 1, 0])
# Define the dog target
dog_target = np.array([0, 0, 0, 0, 0, 0, 1])
# create all the targets
train_cat_targets = np.tile(cat_target, (n_cat_samples, 1))
train_dog_targets = np.tile(dog_target, (n_cat_samples, 1))
targets = np.vstack((train_cat_targets, train_dog_targets))


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

# we add a negative weight between ydog and ycat "by hand"
weights[6, 5] = -0.2


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

# std of noise
sigma = 2.5
# learning rate
alpha = 0.2

# how many test inputs will we use? (n_test/2 for each prototype)
n_test = 10
# use n_test/2 cat_prototype and n_test/2 dog_prototypes as inputs
test_inputs = np.vstack((np.tile(cat_prototype, (int(n_test/2), 1)),
                         np.tile(dog_prototype, (int(n_test/2), 1))))

# we add some noise so that they are all a bit different
test_inputs[:, :5] = test_inputs[:, :5] + np.random.randn(n_test, 5) * samples_noise_std


# array to store the results of the activation optimization
output_result = np.zeros(shape = (n_test, 2))

# loop over all the test inputs
for test_input_n in np.arange(n_test):
    ydog = np.zeros(n_tsteps)
    ycat = np.zeros(n_tsteps)
    
    this_input = test_inputs[test_input_n, :]
    
    incat = weights[5, 0] * this_input[0] + weights[5, 1] * this_input[1] + weights[5, 2] * this_input[2] + weights[5, 3] * this_input[3] + weights[5, 4] * this_input[4]
    indog = weights[6, 0] * this_input[0] + weights[6, 1] * this_input[1] + weights[6, 2] * this_input[2] + weights[6, 3] * this_input[3] + weights[6, 4] * this_input[4]

    # update the ycat value (eq. 2.4)
    ycat[t] = ycat[t-1] + alpha * (incat + weights[6, 5] * ydog[t-1]) + np.random.randn()*sigma
    # update the ydog value (eq. 2.5)
    ydog[t] = ydog[t-1] + alpha * (indog + weights[6, 5] * ycat[t-1]) + np.random.randn()*sigma

    # optimize the activation of the output units
    for t in times[1:]:
        ycat[t] = ycat[t-1] + alpha * (incat + weights[6, 5] * ydog[t-1]) + np.random.randn()*sigma
        ydog[t] = ydog[t-1] + alpha * (indog + weights[6, 5] * ycat[t-1]) + np.random.randn()*sigma
        # rectify the y values: if they are smaller than zero put them at zero
        if ycat[t]<0: ycat[t] = 0
        if ydog[t]<0: ydog[t] = 0
    
    # store the results of the network optimization
    output_result[test_input_n, :] = [ycat[-1], ydog[-1]]



# the np.round function is used to simplify the output, it limits the number
# after the floating point to only one number
print(np.round(output_result, 1))




