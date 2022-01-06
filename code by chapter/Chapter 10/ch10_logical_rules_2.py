#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:24:48 2018

@author: tom verguts
boltzmann machine for implementing logical rules
here, different from ch10_logical_rules.py, a randomly chosen input pattern
is clamped in the free phase  (and again everything in the fixed phase)
"""
#%% import and initialize
import numpy as np
import time
np.set_printoptions(suppress = True, precision = 2)
verbose = 0
n_train_trials = 1000
beta = 0.05 # learning rate
e = 0.001 # small (non-zero) number
T = 1 # temperature
N = 4 # number of units in the network
w = np.zeros((N, N))
patterns = np.array(   [[1,0,0,0], [1,1,0,0], [1,0,1,0], [1,1,1,0],     # patterns = all POSSIBLE patterns
                        [1,0,0,1], [1,1,0,1], [1,0,1,1], [1,1,1,1]] )   # first unit is constant (intercept) 
prob_AND = np.array([  1,       1,       1,       e,
					   e,       e,       e,       1]) # the AND rule
prob_OR = np.array([   1,       e,       e,       e,
				  	   e,       1,       1,       1]) # the OR rule
prob_XOR = np.array([  e,       1,       1,       e,
					   1,       e,       e,       1]) # the XOR rule

prob_fixed = prob_OR # choose your rule here

prob_fixed = prob_fixed/np.sum(prob_fixed) # at pattern level
prob_free = np.ndarray(2**(N-1))      # probability distribution when only part is clamped (and the rest free)
p_fixed = np.ndarray((N, N))   # at feature (and thus matrix) level
p_free  = np.ndarray((N, N))  

def p_from_prob(prob_vector):
    """determine co-occurence probability matrix for cells i and j from probability vector"""
    p_matrix  = np.zeros((N, N))
    for row in range(N):
        for column in range(N):
            som = 0
            for p_index, pattern in enumerate(patterns):
                if (pattern[row]==1) and (pattern[column]==1) :
                    som += prob_vector[p_index]
            p_matrix[row, column] = som
    return p_matrix

def energy(pattern):
    return -np.matmul(np.matmul(pattern,w),pattern)

#%% weight estimation
t1 = time.time()
# fixed phase: check if i and j are jointly active in the required distribution (based on prob)
p_fixed = p_from_prob(prob_fixed)
       
#%% free phase
x = np.ndarray(N)
for loop in range(n_train_trials):
    prob_free = np.zeros(prob_fixed.shape)
    x[0] = 1                        # intercept
    x[1] = np.random.choice([0, 1]) # random clamp
    x[2] = np.random.choice([0, 1]) # random clamp    
    for p_index, pattern in enumerate(patterns):
        prob_free[p_index]  = (pattern[0]==x[0])*\
                              (pattern[1]==x[1])*\
                              (pattern[2]==x[2])*\
             np.exp(-(1/T)*energy(pattern))
    prob_free = prob_free/np.sum(prob_free)      
    p_free = p_from_prob(prob_free)
    w_previous = np.copy(w)
    w = w + beta*(p_fixed - p_free)
    dev = np.sum((p_fixed-p_free)**2, axis = (0,1)) # deviation between p_fixed and p_free, kind of error score
t2 = time.time()
print("simulation took {:.2f} sec".format(t2-t1))

#%% test the network
print("test phase:")
test_patterns = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
for test_pattern in test_patterns:
    for p_index, pattern in enumerate(patterns):
        prob_free[p_index]  = np.all(pattern[0:3]==test_pattern)*\
                              np.exp(-(1/T)*energy(pattern))
    prob_free = prob_free/np.sum(prob_free)
    if verbose:
       print("\n", prob_free)      
    print("for test pattern {} the response is {:.0f}".format(test_pattern, np.floor(np.argmax(prob_free)/4)))      