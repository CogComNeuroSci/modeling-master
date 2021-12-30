#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:57:58 2020

@author: tom verguts
plot column-wise means and standard deviations of a matrix;
for generating table 6.2
note that if each row is an estimate (for a different data set), then the standard deviation is
also a standard error
"""

import numpy as np


v = np.load("simulation_results_50_Powellbayes.npy")
print(np.mean(v, axis=0))
print(np.std(v, axis=0))