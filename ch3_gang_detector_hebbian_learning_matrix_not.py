#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mehdi Senoussi
code adapted by tom verguts to the jets and sharks context
version 1
"""

import numpy as np
import time
from ch0_course_functions import plot_network, update_network
import matplotlib.pyplot as pl

# we have 4 input unit on the first layer and 4 output units on the second layer
layers = np.array([1, 1, 1, 1, 2, 2, 2, 2])
n_units = len(layers)
# here we set all the input activations (index from 0 to 3) to 0.
# We also set the output units to 0.
activations = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

# let's set energy to zero for now
energy = 0


###############################################################################
####    LEARNING PART
###############################################################################

# our learning parameter
beta = .6

# training samples (activation (x) of each input unit)
train_samples = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # john
                          [0, 1, 0, 0, 0, 0, 0, 0],  # paul 
                          [0, 0, 1, 0, 0, 0, 0, 0],  # ringo 
                          [0, 0, 0, 1, 0, 0, 0, 0] ]) # george

# the targets :
targets = np.array(  [    [0, 0, 0, 0, 1, 0, 1, 0], # jet, burglar
                          [0, 0, 0, 0, 1, 0, 1, 0], # jet, burglar
                          [0, 0, 0, 0, 0, 1, 0, 1], # shark, drugsdealer
                          [0, 0, 0, 0, 0, 1, 0, 1], # shark, drugsdealer
                      ] )

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
weights[0, 4, 0] = random_weight # random weight from john
weights[0, 5, 0] = random_weight
weights[0, 6, 0] = random_weight
weights[0, 7, 0] = random_weight
weights[0, 4, 1] = random_weight # random weight from paul
weights[0, 5, 1] = random_weight
weights[0, 6, 1] = random_weight
weights[0, 7, 1] = random_weight
weights[0, 4, 2] = random_weight # random weight from ringo
weights[0, 5, 2] = random_weight
weights[0, 6, 2] = random_weight
weights[0, 7, 2] = random_weight
weights[0, 4, 3] = random_weight # random weight from george
weights[0, 5, 3] = random_weight
weights[0, 6, 3] = random_weight
weights[0, 7, 3] = random_weight



# plot the network to see how it initially looks like
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights[0, :, :], layers = layers, energy = 0)

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    # to wait for any button press to go to the next iteration of the loop
    # you can make this "automatic" by changing the 0 to a number of seconds
    fig.waitforbuttonpress(0)

    weights[trial_n+1, :] = weights[trial_n, :] + \
                                beta * np.dot(targets[trial_n, :][:, np.newaxis], train_samples[trial_n, :][:, np.newaxis].T)
    
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations, change =0,
        unit_pos = unit_pos, weights = weights[trial_n+1, :, :], layers = layers,
        cycle = 0, learn_trial_n = trial_n+1, energy = energy)




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
y_jet   = np.zeros(n_tsteps)
y_shark = np.zeros(n_tsteps)
y_burg  = np.zeros(n_tsteps)
y_drug  = np.zeros(n_tsteps)

# std of noise
sigma = .7
# change rate (scaling parameter)
alpha = .2

# let's add the inhibition between output units
weights_end[4, 5] = -0.3 # jets inhibits sharks
weights_end[5, 4] = -0.3 # cat inhibits dog
weights_end[6, 7] = -0.3 # burglar inhibits drugs dealer
weights_end[7, 6] = -0.3 # drugs dealer inhibits burglar

# let's input a certain pattern of activations (i.e. x1, x2 and x3)
activations[:4] = 1, 0, 0, 0 # activate john

################################
################################
#### USING THE DOT PRODUCT
################################
################################
# computing the initial y activation values
activations = np.dot(weights_end, activations)
################################
################################


in_jet   = activations[4]
in_shark = activations[5]
in_burg  = activations[6]
in_drug  = activations[7]

y_jet[t]   = y_jet[t-1]   + alpha * (in_jet   + weights_end[4, 5] * y_shark[t-1])+ np.random.randn()*sigma
y_shark[t] = y_shark[t-1] + alpha * (in_shark + weights_end[5, 4] * y_jet[t-1])  + np.random.randn()*sigma
y_burg[t]  = y_burg[t-1]  + alpha * (in_burg  + weights_end[6, 7] * y_drug[t-1]) + np.random.randn()*sigma
y_drug[t]  = y_drug[t-1]  + alpha * (in_drug  + weights_end[7, 6] * y_burg[t-1]) + np.random.randn()*sigma

activations[4:] = [y_jet[t], y_shark[t], y_burg[t], y_drug[t]]
energy = (-in_jet*y_jet[t] - in_shark*y_shark[t] - in_burg*y_burg[t] - in_drug*y_drug[t] 
          + weights_end[5, 6]*y_jet[t]*y_shark[t] + weights_end[6, 7]*y_burg[t]*y_drug[t]) 

for t in times[1:]:
    y_jet[t]   = y_jet[t-1]   + alpha * (in_jet   + weights_end[4, 5] * y_shark[t-1])+ np.random.randn()*sigma
    y_shark[t] = y_shark[t-1] + alpha * (in_shark + weights_end[5, 4] * y_jet[t-1])  + np.random.randn()*sigma
    y_burg[t]  = y_burg[t-1]  + alpha * (in_burg  + weights_end[6, 7] * y_drug[t-1]) + np.random.randn()*sigma
    y_drug[t]  = y_drug[t-1]  + alpha * (in_drug  + weights_end[7, 6] * y_burg[t-1]) + np.random.randn()*sigma   
    if y_jet[t] < 0 :
        y_jet[t] = 0
    if y_shark[t] < 0 :
        y_shark[t] = 0
    if y_burg[t] < 0 :
        y_burg[t] = 0    
    if y_drug[t] < 0 :
        y_drug[t] = 0    
    activations[4:] = [y_jet[t], y_shark[t], y_burg[t], y_drug[t]]
    energy = (-in_jet*y_jet[t] - in_shark*y_shark[t] - in_burg*y_burg[t] - in_drug*y_drug[t] 
          + weights_end[4, 5]*y_jet[t]*y_shark[t] + weights_end[6, 7]*y_burg[t]*y_drug[t])
    
    update_network(fig = fig, axs = axs, texts_handles = texts_handles,
        lines_handles = lines_handles, activations = activations,
        unit_pos = unit_pos, weights = weights_end, layers = layers, change = 0,
        cycle = t, energy = energy, learn_trial_n = -1)

    time.sleep(timesleep)