#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:06:01 2020

@author: tom verguts
"""

import numpy as np
import matplotlib.pyplot as plt

ntrial = 100
lrate, ndim = 0.1, 2
w = np.random.randn(2,ntrial)
sigma = np.array([[1, -0.7], [-0.7, 1.0]])
X = np.linalg.cholesky(sigma) # appropriate transformation matrix given covariance sigma
data = np.ndarray((ntrial,ndim))
fig, axs = plt.subplots(1)
for loop in range(1, ntrial):
    sample = np.dot(X,np.random.randn(ndim))
    data[loop] = sample
    y = np.dot(w[:,loop-1],sample)
    axs.scatter(sample[0], sample[1],c = "black")
    #w[:,loop] = w[:,loop-1] + lrate*(sample*y-(y**2)*w[:,loop]) # oja's rule
    w[:,loop] = w[:,loop-1] + lrate*sample*y # explicit normalization
    w[:,loop] = w[:,loop]/np.linalg.norm(w[:,loop])

# plot stuff
stretch = 5
axs.set_xlim(-3, +3)
axs.set_ylim(-3, +3)
axs.set_xlabel("Dimension 1")
axs.set_ylabel("Dimension 2")
axs.plot(stretch*np.array([-w[0, ntrial-1], w[0, ntrial-1]]), stretch*np.array([-w[1, ntrial-1], w[1, ntrial-1]]), c = "black")
print(np.cov(data, rowvar = False)) # to check if data structure is correct; should be similar to sigma