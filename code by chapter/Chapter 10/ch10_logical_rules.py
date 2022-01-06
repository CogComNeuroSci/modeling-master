#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:24:48 2018

@author: tom verguts
Chapter 10, unsupervised learning: boltzmann machine for implementing logical rules
nothing is clamped in the free phase (and everything in the fixed phase)
first unit is a bias unit
approach here is almost literal implementation of Ackley et al (1985, Cog Sci, appendix)
"""
#%% import and initialize
import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress = True, precision = 2)
verbose = 0
n_train_trials = 10000
beta = 0.1 # learning rate
e = 0.01 # small (non-zero) number; to avoid pushing weights to zero
N = 4 # number of units in the network
w = np.zeros((N, N)) # weight matrix; intercept (bias) unit is included
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
            total = 0
            for p_index, pattern in enumerate(patterns):
                if (pattern[row]==1) and (pattern[column]==1) :
                    total += prob_vector[p_index]
            p_matrix[row, column] = total
    return p_matrix

def energy(pattern):
    """energy of a given pattern"""
    return -(np.matmul(np.matmul(pattern,w),pattern) - np.matmul(pattern, np.diag(w)))/2

#%% determine p_fixed
t1 = time.time() # start estimation time
# calculate probability that i and j are jointly active in the required distribution (based on prob_fixed)
p_fixed = p_from_prob(prob_fixed)
       
#%% main weight estimation loop
dev_list = [] # list of trial-by-trial deviations btw p_fixed (target) and p_free (result); it's an error score
x = np.ndarray(N)
for loop in range(n_train_trials):
    T = 4*np.exp(-0.00001*loop) # temperature set via simulated annealing
    for p_index, pattern in enumerate(patterns):
        prob_free[p_index] = np.exp(-(1/T)*energy(pattern))
    prob_free = prob_free/np.sum(prob_free)      
    p_free = p_from_prob(prob_free)
    w_previous = np.copy(w)
    w = w_previous + beta*(p_fixed - p_free)
    dev = np.sum((p_fixed-p_free)**2, axis = (0,1))
    dev_list.append(dev)
t2 = time.time() # end estimation time
print("weight matrix estimation took {:.2f} sec".format(t2-t1))

#%% plot error
fig, ax = plt.subplots()
ax.plot(dev_list)
ax.set_title("deviation btw p_free and p_fixed across time")

#%% test the network on various clamped test patterns
print("test phase:")
test_patterns = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
for test_pattern in test_patterns:
    prob_free = np.zeros(prob_free.shape)
    for p_index, pattern in enumerate(patterns):
        if np.all(pattern[0:3] == test_pattern):
            prob_free[p_index] = np.exp(-(1/T)*energy(pattern))
    prob_free = prob_free/np.sum(prob_free)
    if verbose:
       print("\n", prob_free)      
    print("for test pattern {} the response is {:.0f}".format(test_pattern, np.floor(np.argmax(prob_free)/4)))
#%% test if prob_free approximates prob_fixed
prob_free = np.zeros(prob_free.shape)
for p_index, pattern in enumerate(patterns):
    prob_free[p_index] = np.exp(-(1/T)*energy(pattern))
prob_free = prob_free/np.sum(prob_free)

print("\nfixed probability:")
print(prob_fixed)
print("free probability:")
print(prob_free)      
