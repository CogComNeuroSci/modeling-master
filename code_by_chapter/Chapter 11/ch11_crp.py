#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:07:30 2021

@author: tom verguts
simple example of the discrete-variable chinese restaurant process;
as originally described by Anderson (1991, Psych Rev)
"""
#%% import and initialize
import numpy as np

np.set_printoptions(precision = 1)

n_stim, n_dim = 100, 8
cp = 0.01 # inverse concentration parameter; higher values make it more likely to pick new tables
alpha_par = np.ones(n_dim)/10 # prior parameters
data_random = False # random data with some threshold
if data_random:
    threshold = 0.2 
    data       = (np.random.randn(n_stim, n_dim)>threshold)*1       # random data matrix
else: # clustered data matrix, but with some noise pr
    data       = np.kron(np.eye(2), np.ones((n_stim//2, n_dim//2))) 
    pr = 0.01 # add some noise to data ; 1 = pure random
    data_switch = (np.random.random_sample(np.shape(data)) < pr)*1
    data = data + np.multiply(1-2*data, data_switch)

indx       = np.arange(n_stim)           # order in which stimuli are walked through
assignment = np.zeros((n_stim, n_stim))  # assignment of stimuli to tables
np.random.shuffle(indx)                  # randomly shuffle the stimulus order
n_class = 0
verbose = False 
#%% chinese restaurant process
assignment[indx[0], 0] = 1               # first stimulus to first table
for loop in indx[1:]:
    p = np.ndarray(n_class+1) # proportional to the prob of choosing each class
    for class_loop in range(n_class):
        lik = 1 # the likelihood term
        for dim_loop in range(n_dim):
            prob = ((np.sum(np.multiply(assignment[:,class_loop], data[:,dim_loop])) + alpha_par[dim_loop])
			              /(np.sum(assignment[:,class_loop]) + np.sum(alpha_par))) 
            lik *= np.power(prob, data[loop,dim_loop]) * np.power(1-prob, 1-data[loop,dim_loop])
        prior = np.sum(assignment[:,class_loop])  # this term is proportional to the prior (which is enough to assign the stimulus)	
        p[class_loop] = prior*lik                 # p is proportional to the posterior probability
    p[n_class] = cp 		                      # proportional to the unassigned class probability
    assignment[loop, np.argmax(p)] = 1           # assign the stimulus to the best-fitting class
    n_class += (np.argmax(p) == n_class)         # if assigned to novel class, increase number of classes

#%% report results
print("n classes is {}".format(n_class))
if verbose:
    print(data)
    print(assignment)
	 