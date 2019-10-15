#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi
"""

# this script is divided in two parts
    # 1) a learning part in which we use hebbian learning to train the model, 
    #    i.e change its weights
    #
    # 2) a testing part in which we present cat and dog examplars (vectors of 
    #    input units) to this trained model and use the optimize the energy function.

import numpy as np
import time
from ch0_course_functions import plot_network2, update_network

# first create the structure of the network
# we have 3 input unit on the first layer and 2 output units on the second layer
layers = np.array([1, 1, 1, 2, 2])
n_units = len(layers)

# here we set all the input activations (index from 0, 1 and 2) to 0.
# We also set the two output units ('cat', index 3, and 'dog', index 4) to 0.
activations = np.array([0., 0., 0., 0., 0.])

# let's set energy to zero for now
energy = 0

#%%
###############################################################################
####    LEARNING PART
###############################################################################

# our learning parameter
beta = .8

# learning samples (activation (x) of each input unit)
train_samples = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]])

# the targets (basically representing "dog" or "cat"):
#                      cat        cat        cat        dog        dog        dog
targets = np.array(  [[1, 0],   [1, 0],    [1, 0],    [0, 1],    [0, 1],    [0, 1]])


# how many learning samples do we have (hence, how many trials are we going to do?)
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
randweight = np.random.random()/10.
weights[0, 3:5, :3] = randweight # random weight 


# plot the network to see how it initially looks like
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights[0, :, :], layers = layers, energy = 0)

fig.suptitle('Press any key to do a learning trial')


# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    # just to plot some title indicating what is the sample and the target on
    # on top of the left plot (plot of the network)
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
            old_weight = weights[trial_n, i+3, j]
            
            # what is the change in this particular weight
            delta_weight = beta * x * t
            
            # update our weight for the connection between this
            weights[trial_n +1 , i+3, j] = old_weight + delta_weight
    
    ## Update the plot of the network
    # Here we fixed the parameter "cycle" to 0 because we use it to represent
    # the activation optimization cycle as seen in chapter 2 (which is not what
    # we are doing in this loop).
    # Also we added a parameter "learn_trial_n" which represents the cycle/trial
    # of learning we are in, in other words the learning trial we are using.
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations, change =0,
        unit_pos = unit_pos, weights = weights[trial_n + 1, :, :], layers = layers,
        cycle = 0, learn_trial_n = trial_n+1, energy = energy)
    
    # to wait for any button press to go to the next iteration of the loop
    # you can make this "automatic" by changing the 0 to a number of seconds
    fig.waitforbuttonpress(0)

axs[0].set_title('')

# we add a negative weight between ydog and ycat by hand because the hebbian 
# learning algorithm cannot create it
weights[-1, 4, 3] = -.2

# update the network plot
update_network(fig = fig, axs = axs, texts_handles = texts_handles,
    lines_handles = lines_handles, activations = activations, change =0,
    unit_pos = unit_pos, weights = weights[trial_n+1, :, :], layers = layers,
    cycle = 0, learn_trial_n = trial_n+1, energy = energy)


pl.suptitle('Learning phase finished!\nPress a key to input a certain pattern in the model and see how it behaves!')
fig.canvas.draw()
fig.waitforbuttonpress(0)

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

# let's input a certain pattern of activations (i.e. x1, x2 and x3)
activations[:3] = 1, 1, 0

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
    if ydog[t]<0: ydog[t] = 0
    activations[3:] = [ycat[t], ydog[t]]
    energy = -incat*ycat[t] - indog*ydog[t] - weights_end[4, 3]*ycat[t]*ydog[t]
        
    ## Update the network plot
    # Here we update the parameter "cycle" on every iteration because we use it
    # to represent the activation optimization cycle as seen in chapter 2.
    # The parameter "learn_trial_n" which represents the cycle/trial of
    # learning we are in, is fixed because learning has been done already.
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations,
        unit_pos = unit_pos, weights = weights_end, layers = layers, change = 0,
        cycle = t, energy = energy, learn_trial_n = trial_n+1)

    time.sleep(timesleep)


