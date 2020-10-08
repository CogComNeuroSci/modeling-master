#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:13:24 2019

@author: tom
"""

import numpy as np
from numpy import load

A = load("simulation_results_10_nelder_bayes.npy")
for loop in range(2):
    print(np.mean(A[:,0,loop], axis = 0))
    print(np.mean(A[:,1,loop], axis = 0))
    print(np.mean(A[:,2,loop], axis = 0))
    print(np.std(A[:,0,loop], axis = 0)/np.sqrt(A.shape[0]))
    print(np.std(A[:,1,loop], axis = 0)/np.sqrt(A.shape[0]))
    print(np.std(A[:,2,loop], axis = 0)/np.sqrt(A.shape[0]))
    print(" ")