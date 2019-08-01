#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mehdisenoussi
and tom verguts (learning component)
for the jets & sharks case
"""

import numpy as np
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
layers= np.hstack([np.repeat(1, n_units), np.repeat(2, n_units)])

# let's set energy to zero for now
energy = 0

# random initial weights
weights = np.random.randn(n_units, n_units)

# the previous operation/command also set the "self-connection" to random weights
# if we don't want that, we can fix all units in the diagonal
# at 0 using the identity matrix
weights[np.identity(n_units).astype(np.bool)] = 0
print(weights)

# plot the network to see how it initially looks like
fig, axs, texts_handles, lines_handles, unit_pos =\
    plot_network(figsize = [13, 7], activations = activations,
                  weights = weights, layers = layers, energy = 0)

#%% TRAIN THE MODEL
# training samples
n_trials = 4
train_samples = np.array([[1, 0, 0, 0, 1, 0, 1, 0],  # john
                          [0, 1, 0, 0, 1, 0, 1, 0],  # paul 
                          [0, 0, 1, 0, 0, 1, 0, 1],  # ringo 
                          [0, 0, 0, 1, 0, 1, 0, 1] ]) # george

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

#%% TESTING OUR MODEL
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
