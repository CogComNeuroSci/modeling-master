#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:13:24 2019

@author: tom
"""

import numpy as np
from numpy import load

A = load("simulation_results.npy")
print(np.mean(A[:,:,2], axis = 0))
print(np.mean(A[:,:,3], axis = 0))