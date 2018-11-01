#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi
noisy version by tom verguts
note that I removed all plotting functionality in this exercise; 
all printing goes to the console
"""

import numpy as np

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
train_samples = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]])

# the targets (basically representing "dog" or "cat"):
#                      cat        cat        cat        dog        dog        dog
targets = np.array(  [[1, 0],   [1, 0],    [1, 0],    [0, 1],    [0, 1],    [0, 1]])


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



# we set a weight index because its size is bigger than the number of samples
w_ind = 0

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
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
    

#%%
###############################################################################
####    TESTING PART
###############################################################################
# for testing the effect of noise, changes were only made to this part
# let's only keep the end result of our learning process
print("busy with testing phase...")

weights_end = weights[-1, :, :]


timesleep = .1
n_tsteps = 20
n_test_trials = 200
times = np.arange(n_tsteps)
t = 1

# output units
ydog = np.zeros(n_tsteps)
ycat = np.zeros(n_tsteps)
#incat = np.zeros(n_tsteps)
#indog = np.zeros(n_tsteps)

# std of noise
sigma = np.arange(0, 3, 0.2)
# change rate
alpha = .2

correct_counter = np.zeros(sigma.size) # this collects the number of correct trials

# let's add the inhibition between cat and dog
weights[4, 3] = -0.3 # dog inhibits cat
weights[3, 4] = -0.3 # cat inhibits dog

# let's input a certain pattern of activations (i.e. x1, x2 and x3)
activations[:3] = 1, 1, 0


for noise_loop in range(sigma.size): # loop across all noise levels
    for trial_loop in range(n_test_trials):         # loop across all trials 
        for t in times[1:]:                    # loop across time points in a trial
            incat = weights_end[3, 0] * activations[0] + weights_end[3, 1] * activations[1] + weights_end[3, 2] * activations[2]
            indog = weights_end[4, 0] * activations[0] + weights_end[4, 1] * activations[1] + weights_end[4, 2] * activations[2]
            ycat[t] = ycat[t-1] + alpha * (incat + weights_end[3, 4] * ydog[t-1]) + np.random.randn()*sigma[noise_loop]
            ydog[t] = ydog[t-1] + alpha * (indog + weights_end[4, 3] * ycat[t-1]) + np.random.randn()*sigma[noise_loop]
            if ycat[t]<0 : ycat[t] = 0
            if ydog[t]<0: ydog[t] = 0
            activations[3:] = [ycat[t], ydog[t]]
            energy = -incat*ycat[t] - indog*ydog[t] - weights_end[4, 3]*ycat[t]*ydog[t]
        correct_counter[noise_loop] += (ycat[-1] > ydog[-1])

correct_counter = correct_counter/n_test_trials

# print to the console
for noise_loop in range(sigma.size):
    print("% of correct responses at noise level {0:.1f} = {1:.0%}".format(sigma[noise_loop],correct_counter[noise_loop])) 