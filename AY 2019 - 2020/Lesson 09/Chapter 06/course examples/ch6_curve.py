#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:51:49 2019

@author: tom verguts
show likelihood curve for the learning model
"""

#%% initialize
import numpy as np
from ch6_ST_estimate import generate_learn

learning_rate, temperature = 0.6, 0.4 
low, high, stepsize = -3, +3, .01
x = np.arange(low, high, stepsize)
y = np.arange(low, high, stepsize)

#%% generate some data
generate_learn(alpha = learning_rate, beta = temperature)

# calculate some likelihoods
# calculate logL_learn over a grid of points

#%% plot the grid