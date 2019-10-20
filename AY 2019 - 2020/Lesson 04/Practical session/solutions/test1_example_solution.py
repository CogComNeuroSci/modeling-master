#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi

This script show a solution to the test1_example.
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
from ch0_course_functions import *

timesleep = .1

# first create the structure of the network
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

# our learning parameter is set at 0.1
beta = .1

# learning samples (activation (x) of each input unit)
# 10 cats (i.e. [1, 1, 0]) and 40 dogs (i.e. [-1, 1, 1])

# In the test instructions document the cat prototype was [0, 1, 1] and the dog 
# prototype was [1, 1, -1].
# In this script we changed it to [1, 1, 0] for the cat and [-1, 1, 1] for the
# dog to be more coherent with the previous lessons.
# i.e the first unit is "has its picture on facebook" which is associated with cats, etc.
# The specific prototypes we use are arbitrary, so this doesn't change anything
# to the results and the conclusion the test questions.

# Define the cat prototype
cat_prototype = np.array([1, 1, 0])
# set the number of samples we want to present of this prototype
n_cat_samples = 10
# Define the dog prototype
dog_prototype = np.array([-1, 1, 1])
# set the number of samples we want to present of this prototype
n_dog_samples = 40

# the function tile repeats a certain set of values in the same order a certain
# number of times but it creates a 1-dimensional array of these values so we use
# the method reshape() to rearrange them into an array of n_samples-by-n_values
# finally we use np.vstack() to put the cat and dog samples into the same array
train_samples = np.vstack([np.tile(cat_prototype, n_cat_samples).reshape(n_cat_samples, 3),
                           np.tile(dog_prototype, n_dog_samples).reshape(n_dog_samples, 3)])

# add gaussian noise with mean = 0 and standard deviation = 0.01
# we create an array of size (n_cat_samples + n_dog_samples)-by-3 to add some
# independent gaussian noise to each value in the train_sample array
# we also multiply these values by the standard deviation we want in our samples
samples_noise_std = 0.01
train_samples = train_samples + np.random.randn(n_cat_samples + n_dog_samples, 3) * samples_noise_std

# the targets (basically representing "dog" or "cat"):
# Define the cat target
cat_target = np.array([1, 0])
# Define the dog target
dog_target = np.array([0, 1])
# create the all the targets
targets = np.vstack([np.tile(cat_target, n_cat_samples).reshape([n_cat_samples, 2]),
                     np.tile(dog_target, n_dog_samples).reshape([n_dog_samples, 2])])


# how many learning samples do we have (hence, how many trials are we going to do?)
n_trials = train_samples.shape[0]
# get the number of dimensions (i.e. units) in the samples
n_sample_dim = train_samples.shape[1]
# get the number of dimensions (i.e. units) in the targets
n_target_dim = targets.shape[1]

# create a weight matrix of size n_units-by-n_units
weights = np.zeros(shape = [n_units, n_units])

# let's set a SMALL weight everywhere so that the plotting functions have
# something other than zeros otherwise we won't see the weights in the upper-right plot
weights += 0.01


# plot the network to see how it initially looks like
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights, layers = layers, energy = 0)

# to present the samples in a random order we will create an array with the
# numbers of each trial and shuffle the order of these numbers. Then during the
# training loop across all samples we will use this randomly ordered numbers to
# pick a sample
random_order_trial_number = np.arange(n_trials)
np.random.shuffle(random_order_trial_number)

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    # just to plot some title indicating what is the sample and the target on
    # on top of the left plot (plot of the network)
    axs[0].set_title('this trial: xs = %s, ts = %s' % (train_samples[trial_n, :], targets[trial_n, :]))
    shuffled_index = random_order_trial_number[trial_n]
    # loop on each of the sample's "dimensions" (i.e. each input unit's activation)
    for j in np.arange(n_sample_dim):
        # loop on each of the target's "dimensions" (i.e. each wanted output unit's activation)
        for i in np.arange(n_target_dim):
            # we get this trial xs (i.e. input activations) from our array of samples
            x = train_samples[shuffled_index, j]
            # we get this trial targets (i.e. wanted outputs) from our array of targets
            t = targets[shuffled_index, i]
            
            # get the old weight of this connection
            old_weight = weights[i+3, j]
            
            # what is the change in this particular weight
            delta_weight = beta * x * t
            
            # update our weights
            weights[i+3, j] = old_weight + delta_weight
    
    ## Update the plot of the network
    # Here we fixed the parameter "cycle" to 0 because we use it to represent
    # the activation optimization cycle as seen in chapter 2 (which is not what
    # we are doing in this loop).
    # Also we added a parameter "learn_trial_n" which represents the cycle/trial
    # of learning we are in, in other words the learning trial we are using.
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations, change =0,
        unit_pos = unit_pos, weights = weights, layers = layers,
        cycle = 0, learn_trial_n = trial_n+1, energy = energy)

    # wait a bit to see the changes in weights    
    time.sleep(timesleep)

axs[0].set_title('')

# we add a negative weight between ydog and ycat by hand because the hebbian 
# learning algorithm cannot create it
weights[4, 3] = -.2

# update the network plot
update_network(fig = fig, axs = axs, texts_handles = texts_handles,
    lines_handles = lines_handles, activations = activations, change =0,
    unit_pos = unit_pos, weights = weights, layers = layers,
    cycle = 0, learn_trial_n = trial_n+1, energy = energy)


pl.suptitle('Learning phase finished!\nPress a key to input a certain pattern in the model and see how it behaves!')
fig.canvas.draw()
# wait a bit to see the changes in weights    
time.sleep(.5)

#%%
###############################################################################
####    TESTING PART
###############################################################################


n_tsteps = 20
times = np.arange(n_tsteps)
t = 1

# output units
ydog = np.zeros(n_tsteps)
ycat = np.zeros(n_tsteps)

# std of noise
sigma = .7
# learning rate
alpha = .2

# let's input a certain pattern of activations (i.e. x1, x2 and x3)
#activations[:3] = cat_prototype
activations[:3] = dog_prototype

# computing the initial y activation values
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


#%%
# Question 6. to present the 10 cat-like dogs/dog-like cats we will not plot
# the network create 10 test inputs and get the results (no plotting
# with the course_function module)

n_test = 10
# the cat-like dog we present is the middle between each prototype, so we
# simply compute the average of the two prototype: (cat_prototype + dog_prototype)/2.
cat_like_dog = np.tile((cat_prototype + dog_prototype)/2., n_test).reshape([n_test, 3])

# we add some noise so that they are all a bit different
cat_like_dog = cat_like_dog + np.random.randn(n_test, 3) * samples_noise_std

# array to store the results of the activation optimization
output_result = np.zeros(shape = (n_test, 2))

for test_input_n in np.arange(n_test):
    ydog = np.zeros(n_tsteps)
    ycat = np.zeros(n_tsteps)
    
    activations[:3] = cat_like_dog[test_input_n, :]
    
    incat = weights[3, 0] * activations[0] + weights[3, 1] * activations[1] + weights[3, 2] * activations[2]
    indog = weights[4, 0] * activations[0] + weights[4, 1] * activations[1] + weights[4, 2] * activations[2]
    ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
    ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
    
    for t in times[1:]:
        ycat[t] = ycat[t-1] + alpha * (incat + weights[4, 3] * ydog[t-1]) + np.random.randn()*sigma
        ydog[t] = ydog[t-1] + alpha * (indog + weights[4, 3] * ycat[t-1]) + np.random.randn()*sigma
        if ycat[t]<0 : ycat[t] = 0
        if ydog[t]<0: ydog[t] = 0
        
    output_result[test_input_n, :] = [ycat[-1], ydog[-1]]

print(output_result)
#%%

'''
Question 5.
    I presented a cat and dog prototype to the network and here are the final
    activations of the output units:
        cat_prototype:
            activations[3:]
            Out[26]: array([6.50385376, 4.36559189])
        dog_prototype:
            activations[3:]
            Out[36]: array([ 0., 47.78714242])
    The network seems to reach the correct activations: the cat unit was more
    active when I presented the cat prototype and vice versa for the dog prototype.
    The dog prototype induced a much stronger activation of the dog unit and
    a cat unit activation of zero because the dog prototypes were presented 4
    times more and therefore the weights were larger.

Question 7.
    Results of presenting a cat-like-dog 10 times
        Out[47]: output_result
        array([[ 1.29145752, 18.0973384 ],
               [ 0.        , 26.12607121],
               [ 0.09482967, 21.36550897],
               [ 0.57398879, 19.65877492],
               [ 0.        , 20.29855031],
               [ 0.55513866, 22.67421804],
               [ 0.        , 21.84862372],
               [ 0.        , 21.8505152 ],
               [ 0.        , 21.96029078],
               [ 0.        , 21.02603947]])

    The dog unit is always more active (which we can interpret as the detector 
    detecting a dog) when presenting cat-like-dog.
    This is due to the fact that during training a lot more dog prototypes were
    used as training samples and therefore the weights activating the dog units
    are larger. Another way of saying this is that the detector is biased towards
    detecting dogs.
'''
