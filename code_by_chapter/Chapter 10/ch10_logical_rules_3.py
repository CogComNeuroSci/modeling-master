#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:24:48 2018

@author: tom verguts
boltzmann machine for implementing logical rules
now it's a RESTRICTED boltzmann machine, with hidden units (n_components)
under construction... hyperparameters are not optimized
"""

#%% import and initialize
import numpy as np
from sklearn.neural_network import BernoulliRBM

np.set_printoptions(suppress = True, precision = 4)
n_burnin = 3000
n_test_trials = 10000
X_and = np.array( [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]] )
X_or  = np.array( [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]] )
X_101 = np.array( [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0]] )
X_xor = np.array( [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] )
X = X_or
#X = np.array([[1, 1, 1]])
p = np.zeros((2**X.shape[1],))
p_tot = np.copy(p)
n_samples = 10
n_rep = 5

#%% fit model n_rep times
for rep_loop in range(n_rep):
    p = np.zeros((2**X.shape[1],))
    model = BernoulliRBM(n_components = 5, n_iter = 200, batch_size = 2, learning_rate = 0.1)
    model.fit(X)
    # check equilibrium distribution
    for loop in range(n_test_trials):
        v_old = np.random.choice([0, 1], size = X.shape[1])
        for sample_loop in range(n_samples):
            v_new = model.gibbs(v_old)
            v_old = v_new
        row = np.dot(v_new, 2**np.array([0, 1, 2])) # search for the corresponding row in p (eg (0, 1, 0) --> row 2; (1, 1, 0) --> row 3)
        p[row] += 1
    p = p/np.sum(p)
    p_tot += p     
# end of simulation    
p_tot /= n_rep    

#%% print and plot
print("overall distribution of each X pattern")
print(p_tot)
print("\nconditional distributions of the test patterns")
test_patterns = [[0, 0], [1, 0], [0, 1], [1, 1]]
for test_pattern in test_patterns:
    prob_res = np.zeros((p_tot.shape)) # restricted probabilities
    for p_index, p in enumerate(p_tot):
        prob_res[p_index]  = \
          ((p_index%4) == (test_pattern[0]+test_pattern[1]*2)) * p
    prob_res = prob_res/np.sum(prob_res)
    print(prob_res)      