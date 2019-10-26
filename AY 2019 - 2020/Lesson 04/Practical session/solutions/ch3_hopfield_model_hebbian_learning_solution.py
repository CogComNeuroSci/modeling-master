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
beta = .1

# define the number of units in our hopfield network
# To use the same example as in chapters 2 and 3 here are the "name" of each
# unit by index:
#       0: Mary, 1: Rich, 2: Female, 3: John, 4: Poor, 5: Male
n_unit = 6

# active and inactive unit level
active_unit = 1
inactive_unit = 0

memory_1 = np.array([active_unit, active_unit, active_unit, inactive_unit, inactive_unit, inactive_unit])
n_memory_1_samples = 10
# for question 5) comment the line above and uncomment the line below
#n_memory_1_samples = 30
memory_2 = np.array([inactive_unit, inactive_unit, inactive_unit, active_unit, active_unit, active_unit])
n_memory_2_samples = 10
# for question 5) comment the line above and uncomment the line below
#n_memory_2_samples = 5

# make multiple examples using the 
train_samples = np.vstack([np.tile(memory_1, n_memory_1_samples).reshape([n_memory_1_samples, n_unit]),
                          np.tile(memory_2, n_memory_2_samples).reshape([n_memory_2_samples, n_unit])])


# how many samples
n_trials = train_samples.shape[0]
weight = np.zeros(shape=[n_unit, n_unit])

# loop over all samples/trials to train our model
for trial_n in np.arange(n_trials):
    # hebbian learning on hopfield model using dot product
    weight = weight + beta * np.dot(train_samples[trial_n, :][:, np.newaxis],
                                    train_samples[trial_n, :][:, np.newaxis].T)

    # the previous operation/command also set the "self-connection" to -1, we don't
    # want that so we say that all units in the diagonal are at 0 using the identity
    # matrix
    weight[np.identity(n_unit).astype(np.bool)] = 0    



#%% Optimization of the network energy (eq. 2.6)

# How many steps are we taking before we stop trying to optimize the network
max_n_step = 30
# the threshold activity level to switch a unit (active->inactive or inactive-active)
threshold = np.ones(n_unit)/2

# the threshold of deviance of the network (sum of differences of all units
# at a certain optimization step). If the deviance of the network is below that
# threshold we know the units' activity levels will not change anymore so we
# can stop the optimization.
stop_threshold = 0.5

# just as a reminder here is what each index represent:
# 0: Mary, 1: Rich, 2: Female, 3: John, 4: Poor, 5: Male

# for question 3)
# We will give our network 2 incomplete input: 0: [John, Male], 1: [Mary, Female]
x_test = np.vstack([np.array([0, 0, 0, 1, 0, 1]), # [John, Male]
                   np.array([1, 0, 1, 0, 0, 0])]) # [Mary, Female]

# for question 4)
# We will give our network 2 incomplete mixed input:
#    0: [Mary, Female, Poor], 1: [Rich, John, Male]
    
#x_test = np.vstack([np.array([1, 0, 1, 0, 1, 0]), # [Mary, Female, Poor]
#                   np.array([0, 1, 0, 1, 0, 1])]) # [Rich, John, Male]
    
    
# for question 6)
# We generate 20 random inputs (array of six 0s or 1s)
# To do this we create an array of size 20-by-6 (20 inputs, each of size 6
#   because we have 6 units) using the function np.random.random_sample().
#   This function draws numbers between 0 and 1 from uniform distribution.
#   We then threshold these numbers (if they are below 0.5 -> 0, above 0.5 -> 1)
#   Because we get booleans when we threshold like this, we then need to transform
#   these numbers into integers (from True and False to 0 and 1) using the array.astype(np.int) method
    
#x_test = (np.random.random_sample(size = [20, 6])>.5).astype(np.int)

# initialize
n_trials = x_test.shape[0]

for loop in range(n_trials):
    # start
    x = x_test[loop, :]
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
    


#%%
    
'''
3) when presenting two incomplete inputs ([John, Male] and [Mary, Female]) the
    network converges towards the correct memory, i.e. respectively memory_2 and
    memory_1:
        
        start:[0 0 0 1 0 1]
        [0 0 0 1 1 1]
        [0 0 0 1 1 1]
                ->stop criterion reached
        
        start:[1 0 1 0 0 0]
        [1 1 1 0 0 0]
        [1 1 1 0 0 0]
                ->stop criterion reached
                
4) When presenting incomplete mixed inputs the network converges toward *both*
    memories active! This is because the activity of the training samples are
    made of 0s and 1s, therefore there is no negative weight in the network
    and if just 1 unit of a memory is active it will activate the others but
    more importantly even if 2 units of another memory are active they will not
    inhibit the other memory from being reinstated:
    
        start:[1 0 1 0 1 0]
        [1 1 1 1 0 1]
        [1 1 1 1 1 1]
        [1 1 1 1 1 1]
                ->stop criterion reached
        
        start:[0 1 0 1 0 1]
        [1 0 1 1 1 1]
        [1 1 1 1 1 1]
        [1 1 1 1 1 1]
                ->stop criterion reached
    
    -> Check for yourselves! use 1 and -1 for the levels of active and inactive
        units in the training samples and see whether this allows the network
        to converge to only one memory when presented with incomplete mixed
        inputs
        
5) Same thing as in question 4). The network converges towards both memories
    active because there is no inhibition between memories:
    
        start:[1 0 1 0 1 0]
        [1 1 1 1 0 1]
        [1 1 1 1 1 1]
        [1 1 1 1 1 1]
                ->stop criterion reached
        
        start:[0 1 0 1 0 1]
        [1 0 1 1 1 1]
        [1 1 1 1 1 1]
        [1 1 1 1 1 1]
                ->stop criterion reached
    
6) Again, because we do not have inhibitory weights most of the random inputs
    converge towards all memories being active since if any unit of a memory
    is active, the other will become active too.
    -> Check what happens if you use 1 and -1 for the levels of active and inactive
        units in the training samples. This should lead to a different result
        in which the most presented memory during the training will be more often
        reached if you present random inputs because its weights will be larger.
    
    
    
    
    
    