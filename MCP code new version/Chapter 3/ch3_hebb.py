#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:35:15 2020

@author: tom
"""
import numpy as np
v1 = np.array([-0.5, 0.1, -0.7])
v2 = np.array([0.6, 0.3, 1.2])
W1 = np.dot(v1[:,np.newaxis], v2[np.newaxis,:])
print(W1)
W2 = np.dot(v2[:,np.newaxis], v1[np.newaxis,:])
print(W2)