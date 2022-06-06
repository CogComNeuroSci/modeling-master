#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:06:01 2020

@author: tom verguts
chapter 10: unsupervised learning: Unsupervised Hebbian learning rule
Check if weights converge to first eigenvector of covariance matrix (principal component (pc))
Note: statement in MCP book that implicit (Oja) normalization is often not enough
to keep weights within reasonable bounds, is not correct. Implicit normalization usually works well
(just like explicit normalization does)
"""

#%% import and initialize
import numpy as np
import matplotlib.pyplot as plt

n_trials = 100
lrate, n_dims = 0.1, 2
w = np.random.randn(2,n_trials)
sigma = np.array([[1.0, 0.7], [0.7, 1.0]]) # covariance matrix
eigval, eigvec = np.linalg.eig(sigma)
pc = eigvec[:, np.argmax(eigval)]          # first principal component, explicitly computed
X = np.linalg.cholesky(sigma)           # appropriate transformation matrix given covariance sigma
data = np.ndarray((n_trials,n_dims))
normalize_explicit, normalize_implicit = False, True # you can normalize explicitly and implicitly independent of each other
fig, axs = plt.subplots(1)

#%% main algorithm
for loop in range(1, n_trials):
    sample = np.dot(X,np.random.randn(n_dims)) # give sample the correct covariance structure
    data[loop] = sample
    y = np.dot(w[:,loop-1],sample) # eq (10.1)
    axs.scatter(sample[0], sample[1],c = "black")
    if normalize_implicit:
        w[:,loop] = w[:,loop-1] + lrate*(sample*y-(y**2)*w[:,loop-1]) # oja's rule, eq (10.3)
    else:		
        w[:,loop] = w[:,loop-1] + lrate*sample*y                      # standard hebb rule, eq (10.2)
    if normalize_explicit:
        w[:,loop] = w[:,loop]/np.linalg.norm(w[:,loop])               # explicit normalization

#%% print and plot results
stretch = 5
axs.set_xlim(-3, +3)
axs.set_ylim(-3, +3)
axs.set_xlabel("Dimension 1")
axs.set_ylabel("Dimension 2")
axs.plot(stretch*np.array([-w[0, n_trials-1], w[0, n_trials-1]]), stretch*np.array([-w[1, n_trials-1], w[1, n_trials-1]]), c = "black")
axs.plot(stretch*np.array([-pc[0], pc[0]]), stretch*np.array([-pc[1], pc[1]]), c = "red")
axs.set_title("black line is principal component (pc) estimated by model; \nred line is analytically calculated pc")
print(np.cov(data, rowvar = False)) # to check if data structure is correct; should be similar to sigma