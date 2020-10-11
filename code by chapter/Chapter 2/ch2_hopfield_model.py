#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:15:46 2017

@author: tom, mehdi
Hopfield model

Instructions for the exercise:
    This network's weights are set by hand between line 32 and 41 for now.
    The activity of the units in this network can only be 0 (inactive) or 1 (active)

    1)  Create a set of training samples to train the network to have 2 memories:
        memory_1: {Mary, Rich, Female} and memory 2: {John, Poor, Male}.
        Create 10 examples of each memory to train the network.

    2)  Train the network using hebbian learning and the dot product. Don't
        forget to subtract the identity matrix to not have units being
        connected to themselves.
        
    3)  Test the network by presenting an incomplete version of each memory and
        check that each incomplete memory converges to the correct complete memory.
        
    4)  Test the network by presenting an "incomplete mixed" inputs such as
        [Mary, Female, Poor] or [Rich, John, Poor] for example.
        Check which memory the network converges to.
        
    5)  Now train the network with 30 examples of memory_1 and 5 of
        memory_2. Finally test the network with "incomplete mixed"
        examples like in question 4). What happens? Do you understand why?
    
    6)  Using the network trained on 30 examples of memory_1 and 5 of memory_2
        and present 20 RANDOM inputs (array of size 6 of zeros and ones) to the
        network.
        How many times does it converges to memory_1? To memory 2? Can you
        explain why this happens?

"""

# import
import numpy as np

# the value for the different weights of the network
w_p = 1 # plus one
w_m = -1 # minus one

# define the number of units in our hopfield network
# To use the same example as in chapters 2 and 3 here are the "name" of each
# unit by index:
#       0: Mary, 1: Rich, 2: Female, 3: John, 4: Poor, 5: Male
n_unit = 6

# create the weight matrix of the network. All units are connected to all units.
weight = np.zeros((6,6))
# set the self-connections to 1
for loop in range(n_unit):
    weight[loop,loop] = w_p

# implement the different memories by hand: put a positive weight between units
# representing a memory and a negative weight between memories
weight[1,:1] = [w_p]
weight[2,:2] = [w_p, w_p]
weight[3,:3] = [w_m, w_m, w_m]
weight[4,:4] = [w_m, w_m, w_m, w_p]
weight[5,:5] = [w_m, w_m, w_m, w_p, w_p]
# here, I make the matrix symmetric
weight = weight + weight.transpose() - np.diagflat(np.diag(weight))
# The two memories stored in this network are therefore:
#   {Mary, Rich, Female}:   [1, 1, 1, 0, 0, 0]
#   {John, Poor, Male}:     [0, 0, 0, 1, 1, 1]


#%% Optimization of the network energy (eq. 2.6)

# initialize
n_trials = 5

# How many steps are we taking before we stop trying to optimize the network
max_n_step = 30
# the threshold activity level to switch a unit (active->inactive or inactive-active)
threshold = np.ones(n_unit)/2

# the threshold of deviance of the network (sum of differences of all units
# at a certain optimization step). If the deviance of the network is below that
# threshold we know the units' activity levels will not change anymore so we
# can stop the optimization.
stop_threshold = 0.5

# We will give our network 2 incomplete memories: 0: [John, Male], 1: [Mary, Female]
x_test = np.vstack([np.array([0, 0, 0, 1, 0, 1]), # [John, Male]
                   np.array([1, 0, 1, 0, 0, 0])]) # [Mary, Female]
                   
for loop in range(n_trials):
    # start
	# here, I sample a random testing pattern
	nr = np.random.randint(x_test.shape[0])
	x = x_test[nr, :]
	print("\nstart:{}".format(x))
    # initialize the counter to do not do more than max_n_steps
	counter = 0
    # initialize stop_crit to False
	stop_crit = False
    # while we haven't reached a stop_criterion and we're still below max_n_steps of optimization
	while not stop_crit and counter<max_n_step:
        # compute the new activation using the weights and a dot product
        # since our units can only be active (1) or inactive (0) we use the 
        # threshold to convert unit activations to 0s and 1s
		x_new = np.array(np.dot(weight, x) > threshold, dtype=int)
        # compute the deviance (how much the network changed after this
        # optimization step)
		deviance = np.sum(np.abs(x-x_new))
        # check that the deviance is not below the strop_threshold
		if deviance<stop_threshold:
			stop_crit = True
        # increment the counter of steps
		counter += 1
		x = x_new
		print(x)
        
    # if we reached the stop criterion for that trial
	if stop_crit: 
		crit_string = ""
    # if we have not reached the stop criterion for that trial, meaning that
	    # the network did not converge toward a stable memory in max_n_steps steps.
	else: 
		crit_String = "not "
	print("\t->stop criterion " + crit_string + "reached")
    
