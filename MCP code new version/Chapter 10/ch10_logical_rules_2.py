#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:24:48 2018

@author: tom verguts
boltzmann machine for implementing logical rules
here, the input pattern is clamped in the free phase 
(and everything in the fixed phase)
works more robustly than direct approach in ch11_logical_rules.py
"""
#%% initialize
import numpy as np
import time
np.set_printoptions(suppress = True, precision = 2)
n_train_trials = 1000
beta = 0.05
e = 0.001 # small (non-zero) number
T = 1 # temperature
N = 4 # number of units in the network
w = np.zeros((N, N))
patterns = np.array(   [[1,0,0,0], [1,1,0,0], [1,0,1,0], [1,0,0,1],
                        [1,1,1,0], [1,1,0,1], [1,0,1,1], [1,1,1,1]] )
prob_fixed = np.array([  1,       1,       1,       e,
                         e,       e,       e,       1]) # the AND rule
#prob_fixed = np.array([  1,       e,       e,       e,
#                         e,       1,       1,       1]) # the OR rule
#prob_fixed = np.array([  e,       1,       1,       1,
#                         e,       e,       e,       1]) # the XOR ruleprob_fixed = prob_fixed/np.sum(prob_fixed) # at pattern level
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
print(p_fixed) 
       
#%% main loop
x = np.ndarray(N)
for loop in range(n_train_trials):
    x[0] = 1
    x[1] = np.random.choice([0, 1]) # random clamp
    x[2] = np.random.choice([0, 1]) # random clamp    
    for p_index in range(len(patterns)):
        prob_free[p_index]  = (patterns[p_index][0]==x[0])*\
                              (patterns[p_index][1]==x[1])*\
                              (patterns[p_index][2]==x[2])*\
             np.exp( (1/T)*np.matmul(np.matmul(patterns[p_index],w),patterns[p_index]) )
         
    prob_free = prob_free/np.sum(prob_free)      
    for row in range(N):
        for column in range(N):
            som = 0
            for p_index in range(len(patterns)):
                if (patterns[p_index][row]==1) and (patterns[p_index][column]==1) :
                    som += prob_free[p_index]
            p_free[row, column] = som
    w_previous = np.copy(w)
    w = w + beta*(p_fixed - p_free)
    dev = np.sum((w-w_previous)**2, axis = (0,1))
    #print(dev)
print(w)
print(prob_free)
t2 = time.time()

#%% test the network
print("simulation took {:.2f} sec".format(t2-t1))
print("test phase:")
test_patterns = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
for test_index in range(len(test_patterns)):
    for p_index in range(len(patterns)):
        prob_free[p_index]  = np.all(patterns[p_index][0:3]==test_patterns[test_index])*\
                              np.exp( (1/T)*np.matmul(np.matmul(patterns[p_index],w),patterns[p_index]) )
    prob_free = prob_free/np.sum(prob_free)
    print(prob_free)      