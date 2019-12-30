#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:24:48 2018

@author: tom verguts
boltzmann machine for implementing logical rules
nothing is clamped in the free phase (and everything in the fixed phase)
first unit is a bias unit
approach here is almost literal implementation of Ackley et al (1985, Cog Sci, appendix)
"""
#%% initialize
import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress = True, precision = 2)
n_train_trials = 10000
beta = 0.1 # learning rate
e = 0.01 # small (non-zero) number
N = 4 # number of units in the network
w = np.zeros((N, N))
patterns = np.array(   [[1,0,0,0], [1,1,0,0], [1,0,1,0], [1,1,1,0],     # patterns = all POSSIBLE patterns
                        [1,0,0,1], [1,1,0,1], [1,0,1,1], [1,1,1,1]] )
#prob_fixed = np.array([  1,       1,       1,       e,
#                         e,       e,       e,       1]) # the AND rule
prob_fixed = np.array([  1,       e,       e,       e,
                         e,       1,       1,       1]) # the OR rule
#prob_fixed = np.array([  e,       1,       1,       1,
#                         e,       e,       e,       1]) # the XOR rule

prob_fixed = prob_fixed/np.sum(prob_fixed) # at pattern level
prob_free = np.ndarray(2**(N-1))      # probability distribution when only part is clamped (and the rest free)
p_fixed = np.ndarray((N, N))   # at feature (and thus matrix) level
p_free  = np.ndarray((N, N))  
t1 = time.time()
# check if i and j are jointly active in the required distribution (based on prob)
for row in range(N):
    for column in range(N):
        som = 0
        for p_index in range(len(patterns)):
            if (patterns[p_index][row]==1) and (patterns[p_index][column]==1) :
                som += prob_fixed[p_index]
        p_fixed[row, column] = som
       
#%% main loop
dev_list = []
x = np.ndarray(N)
for loop in range(n_train_trials):
    T = 4*np.exp(-0.00001*loop) # temperature set via simulated annealing
    for p_index in range(len(patterns)):
        energy = \
         (np.matmul(np.matmul(patterns[p_index],w),patterns[p_index]) - np.matmul(patterns[p_index], np.diag(w)))/2
        prob_free[p_index] = np.exp( (1/T)*energy )
    prob_free = prob_free/np.sum(prob_free)      
    for row in range(N):
        for column in range(N):
            som = 0
            for p_index in range(len(patterns)):
                if (patterns[p_index][row]==1) and (patterns[p_index][column]==1) :
                    som += prob_free[p_index]
            p_free[row, column] = som
    w_previous = np.copy(w)
    w = w_previous + beta*(p_fixed - p_free)
    dev = np.sum((p_fixed-p_free)**2, axis = (0,1))
    dev_list.append(dev)
print(w)
print(prob_free)
t2 = time.time()

#%% test the network
print("simulation took {:.2f} sec".format(t2-t1))
print("test phase:")
prob_free = np.zeros(prob_free.shape)
test_patterns = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
for test_index in range(len(test_patterns)):
    for p_index in range(len(patterns)):
        if np.all(patterns[p_index][0:3]==test_patterns[test_index]):
            energy = (np.matmul(np.matmul(patterns[p_index],w),patterns[p_index]) - np.matmul(patterns[p_index], np.diag(w)))/2
            prob_free[p_index] = np.exp( (1/T)*energy )
    prob_free = prob_free/np.sum(prob_free)
    print(prob_free)      
plt.plot(dev_list)

#%% test if he approximates prob_fixed
prob_free = np.zeros(prob_free.shape)
for p_index in range(len(patterns)):
    energy = (np.matmul(np.matmul(patterns[p_index],w), patterns[p_index]) - np.matmul(patterns[p_index], np.diag(w)))/2
    prob_free[p_index]  += np.exp( (1/T)*energy )
prob_free = prob_free/np.sum(prob_free)
print(prob_free)      