#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 18:11:31 2021

@author: tom verguts
simple online updater
can be used to check convergence properties etc
"""

import numpy as np
import matplotlib.pyplot as plt

n_trials = 100
beta, gamma = 0.5, 1
mean, std = 1, 0.00

w = np.ndarray(n_trials)

for loop in range(n_trials):
	R = np.random.randn()*std + mean
	w[loop] = w[loop-(loop>1)] + beta*(R - gamma*w[loop-(loop>1)])

plt.plot(w)