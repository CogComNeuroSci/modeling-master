#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi
code adapted by tom verguts for test1
this is where code is written to the console
"""

import numpy as np
from ch0_course_functions import plot_network, update_network
import matplotlib.pyplot as pl

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

# our learning parameter
beta = .05

# training samples (activation (x) of each input unit)
cat_proto = [0, 1, 1]
n_train_cats = 30
dog_proto = [1, 1, -1]
n_train_dogs = 40
std_noise = 0.01

train_samples = cat_proto
for loop in range(n_train_cats-1):
    train_samples = np.vstack((train_samples, cat_proto))
for loop in range(n_train_dogs):
    train_samples = np.vstack((train_samples, dog_proto))
    
noise = np.random.randn(n_train_cats+n_train_dogs, 3)*std_noise
train_samples = train_samples + noise

# the targets (basically representing "dog" or "cat"):
#                      cat  
targets = np.array( [1, 0] )
for loop in range(n_train_cats-1):
    targets = np.vstack((targets, [1, 0]))
for loop in range(n_train_dogs):
    targets = np.vstack((targets, [0, 1]))

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
# random.random() yields a number between 0 and 1, then we divide by 50 so we
# get a number between 0 and 0.02
weights[0, 3, 0] = np.random.random()/50. # random weight 
weights[0, 4, 0] = np.random.random()/50. # dogs rarely bite visitors
weights[0, 3, 1] = np.random.random()/50. # cats often have four legs
weights[0, 4, 1] = np.random.random()/50. # dogs often have four legs
weights[0, 3, 2] = np.random.random()/50. # cats rarely have their pictures on FB
weights[0, 4, 2] = np.random.random()/50. # dogs often have their pictures on FB


# plot the network to see how it initially looks like
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights[0, :, :], layers = layers, energy = 0)

fig.suptitle('Press any key to do a training trial')

# we set a weight index because its size is bigger than the number of samples
w_ind = 0

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    axs[0].set_title('this trial: xs = %s, ts = %s' % (train_samples[trial_n, :], targets[trial_n, :]))
    # loop on each of the sample's "dimensions" (i.e. each input unit's activation)
    for j in np.arange(n_sample_dim):
        # loop on each of the target's "dimensions" (i.e. each wanted output unit's activation)
        for i in np.arange(n_target_dim):
            # we get this trial xs (i.e. input activations) from our array of samples
            x = train_samples[trial_n, j]
            # we get this trial targets (i.e. wanted outputs) from our array of targets
            t = targets[trial_n, i]
            
            # get the old weight of this connection
            old_weight = weights[w_ind, i+3, j]
            
            # what is the change in this particular weight
            delta_weight = beta * x * t
            
            # update our weight for the connection between this
            weights[w_ind + 1, i+3, j] = old_weight + delta_weight
    w_ind += 1
    
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations, change =0,
        unit_pos = unit_pos, weights = weights[w_ind, :, :], layers = layers,
        cycle = 0, learn_trial_n = trial_n+1, energy = energy)
    
    # to wait for any button press to go to the next iteration of the loop
    # you can make this "automatic" by changing the 0 to a number of seconds
    #fig.waitforbuttonpress(0)

axs[0].set_title('')


#%%
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

# define test cases
n_test= 2 + 3
new_cat = np.array(cat_proto)
new_dog = np.array(dog_proto)
std_noise_test = 0.01

test_sample = np.zeros((n_test,3))
test_sample[0, :] = new_cat
test_sample[1, :] = new_dog
test_sample[2, :] = (new_cat + new_dog)/2 # this is a cat-like dog (or dog-like cat)
for i in range(3):
    test_sample[i,:] = test_sample[2,:]

test_sample = test_sample + np.random.randn(n_test, 3)*std_noise_test

test_output = np.zeros((n_test,2))
    
for test_loop in range(n_test):

    # let's input a certain pattern of activations (i.e. x1, x2 and x3)
    activations[:3] = test_sample[test_loop,:]

    # computing the initial y activation values
    incat = weights_end[3, 0] * activations[0] + weights_end[3, 1] * activations[1] + weights_end[3, 2] * activations[2]
    indog = weights_end[4, 0] * activations[0] + weights_end[4, 1] * activations[1] + weights_end[4, 2] * activations[2]
    ycat[t] = ycat[t-1] + alpha * (incat + weights_end[4, 3] * ydog[t-1]) + np.random.randn()*sigma
    ydog[t] = ydog[t-1] + alpha * (indog + weights_end[4, 3] * ycat[t-1]) + np.random.randn()*sigma
    activations[3:] = [ycat[t], ydog[t]]
    energy = -incat*ycat[t] - indog*ydog[t] - weights_end[4, 3]*ycat[t]*ydog[t]
    
    for t in times[1:]:
        ycat[t] = ycat[t-1] + alpha * (incat + weights_end[3, 4] * ydog[t-1]) + np.random.randn()*sigma
        ydog[t] = ydog[t-1] + alpha * (indog + weights_end[4, 3] * ycat[t-1]) + np.random.randn()*sigma
        if ycat[t]<0 : ycat[t] = 0
        if ydog[t]<0 : ydog[t] = 0
        activations[3:] = [ycat[t], ydog[t]]
        energy = -incat*ycat[t] - indog*ydog[t] - weights_end[4, 3]*ycat[t]*ydog[t]
    print("output for test {0} equals {1}".format(test_loop, activations[3:]))    