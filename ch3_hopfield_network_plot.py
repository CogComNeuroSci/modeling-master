#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mehdisenoussi
"""

import numpy as np
import time
from ch0_course_functions import plot_network, update_network

# here we set all the activations to 0.
# This is how we represent the hopfield network from the course:
# index 0: Mary
# index 1: John
# index 2: Female
# index 3: Male
# index 4: Rich
# index 5: Poor
activations = np.array([0., 0., 0., 0., 0., 0.])
n_units = len(activations)

# let's set energy to zero for now
energy = 0

weights = np.zeros(shape=[n_units, n_units])

pos_weight = 1/5
neg_weight = -1/5
# let's set some weights so that it mimicks what we saw in the course: there
# are two memories in this network {Mary, rich, female} and {John, poor, male}
# Mary and Female are positively connected
weights[2, 0], weights[0, 2] = pos_weight, pos_weight
# Mary and Rich are positively connected
weights[4, 0], weights[0, 4] = pos_weight, pos_weight
# Rich and Female are positively connected
weights[4, 2], weights[2, 4] = pos_weight, pos_weight
# John and Male are positively connected
weights[1, 3], weights[3, 1] = pos_weight, pos_weight
# John and Poor are positively connected
weights[1, 5], weights[5, 1] = pos_weight, pos_weight
# Male and Poor are positively connected
weights[3, 5], weights[5, 3] = pos_weight, pos_weight

# all the others weights (which represent the connection between units
# belonging to the two different memories) are set to -1
weights[weights==0] = neg_weight

# the previous operation/command also set the "self-connection" to -1, we don't
# want that so we say that all units in the diagonal are at 0 using the identity
# matrix
weights[np.identity(n_units).astype(np.bool)] = 0

#%%
## TESTING OUR MODEL
n_trials = 1
threshold = .1
stop_threshold = .001
max_n_step = 50

for trial_n in range(n_trials):
    # create a random vector of activations to input in our model
    x = np.random.randint(low=0, high=2, size = n_units) # random starting pattern
    # we store the starting vector of activations to show it in the figure title
    x_orig = x
    # print this vector
    print("\nstart:\t\t\t\t\t{}".format(x))
    # set the counter to zero
    counter = 0
    # set the stop criterion to False to start the loop
    stop_crit = False
    
    ######### PLOT THE NETWORK WITH THE COURSE FUNCTIONS
    # we used a trick to plot the evolution of a hopfield network using the
    # course functions. If this looks too complicated to you, don't mind it,
    # it's not important for the course.
    activations = np.hstack([x, np.zeros(n_units)])
    
    if trial_n == 0:
        layers = np.hstack([np.repeat(1, n_units), np.repeat(2, n_units)])
        weights_plot = np.zeros([n_units*2, n_units*2], dtype=np.int)
        weights_plot[n_units:, :n_units] = weights
        fig, axs, texts_handles, lines_handles, unit_pos =\
            plot_network(figsize = [13, 7], activations = activations,
                          weights = weights_plot, layers = layers)
        axs[0].set_title('trial num: %i, x = %s' % (trial_n, np.str(x_orig)))
    else:
        ######### UPDATE THE NETWORK
        axs[0].set_title('trial num: %i, x = %s' % (trial_n, np.str(x_orig)))
        update_network(fig = fig, axs = axs, texts_handles = texts_handles,
            lines_handles = lines_handles, activations = activations,
            unit_pos = unit_pos, weights = weights_plot, layers = layers, change = 0,
            cycle = counter, learn_trial_n = trial_n)

    #########
    
    # loop to optimize our model
    while not stop_crit and counter < max_n_step:
        # compute the activations of our model using the weights and the input
        # vector (the dot product). Then we compare the activations to the
        # threshold, if a unit i activated above the threshold then the
        # operation "np.dot(weights, x) > threshold" is True. We finally say
        # "transform all these True and False into integers which means
        # True -> 1 and False -> 0
        x_new = np.array(np.dot(weights, x) > threshold, dtype = int)
        
        
        
        ######### UPDATE THE NETWORK
        activations = np.hstack([x, x_new])
        axs[0].set_title('trial num: %i, x = %s' % (trial_n, np.str(x_orig)))        
        update_network(fig = fig, axs = axs, texts_handles = texts_handles,
            lines_handles = lines_handles, activations = activations,
            unit_pos = unit_pos, weights = weights_plot, layers = layers, change = 0,
            cycle = counter, learn_trial_n = trial_n)

        # to wait for any button press to go to the next iteration of the loop
        # you can make this "automatic" by changing the 0 to a number of seconds
        fig.waitforbuttonpress(0)
        #########
        
        
        # computes whether the activations changed enough for us to continue
        # the optimization. We use a certain threshold (stop_theshold) which
        # represents a value at which we decide that it is not worth it to
        # continue optimizing the network because it will stay in that state.
        deviance = np.sum(np.abs(x-x_new))
        if deviance < stop_threshold:
            stop_crit = True
        
        counter += 1
        
        # now x (the activation of the network) is set to the new activation
        # we computed using the optimization formula so that at the next
        # iteration we start from there.
        x = x_new
        
        # print the new model activations
        print("\tmodel activations step {0}:\t{1}".format(counter, x_new))
        

        

    # print whether we achieved the stop criterion (meaning that the model will not
    # evolve anymore) or if we had to stop optimizing because we already did the
    # maximum number of optimization steps we set.
    if stop_crit:
        crit_string = ""
    else:
        crit_String = "not "
    print("\t\t\t-> stop criterion " + crit_string + "reached")













