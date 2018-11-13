#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:44:26 2018

@author: tom verguts
illustration of the softmax rule

"""
import numpy as np
w = np.array([2, 0.2, 0.5, 1])
gamma = [0.2, 2] # low and high value
for gamma_loop in gamma:
    denominator = np.sum(np.exp(gamma_loop*w))
    print( np.exp(gamma_loop*w)/denominator )