#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:24:48 2018

@author: tom verguts
boltzmann machine for implementing logical rules
now it's a RESTRICTED boltzmann machine
doesn't work for nonlinear mappings; I have no idea why
"""

#%% initialize
import numpy as np
from sklearn.neural_network import BernoulliRBM
np.set_printoptions(suppress = True, precision = 2)
n_burnin = 3000
n_test_trials = 10000
X_and = np.array( [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]] )
X_or  = np.array( [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]] )
X_101 = np.array( [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0]] )
X_xor = np.array( [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] )
X = X_and
p = np.zeros((2**X.shape[1],))
p_tot = np.copy(p)
n_rep = 5

#%% fit model n_rep times
for rep_loop in range(n_rep):
    p = np.zeros((2**X.shape[1],))
    model = BernoulliRBM(n_components = 5, n_iter = 10000, batch_size = 1, learning_rate = .2)
    model.fit(X)

    # check equilibrium distribution
    v_old = np.random.choice([0, 1], size = X.shape[1])
    for loop in range(n_test_trials):
        v_new = model.gibbs(v_old)
        v_old = v_new
        if loop>n_burnin:
            row = np.dot(v_new, 2**np.array([0, 1, 2]))
            p[row] += 1
    p = p/np.sum(p)
    p_tot += p     
# end of simulation    

#%% wrap up
p_tot /= n_rep    
print("overall distribution")
print(p_tot)
print("conditional distributions")
test_patterns = [[0, 0], [1, 0], [0, 1], [1, 1]]
for test_index in range(len(test_patterns)):
    prob_res = np.zeros((p_tot.shape)) # restricted probabilities
    for p_index in range(len(p_tot)):
        prob_res[p_index]  = \
          ((p_index%4) == (test_patterns[test_index][0]+test_patterns[test_index][1]*2)) * p_tot[p_index]
    prob_res = prob_res/np.sum(prob_res)
    print(prob_res)      