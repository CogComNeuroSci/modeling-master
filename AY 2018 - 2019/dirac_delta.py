#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:19:58 2019

@author: tom verguts
checking how dirac delta function behaves
"""

import numpy as np
import matplotlib.pyplot as plt

low, high = 0.00001, 1
n_support = 100
delta = np.linspace(high, low, num = n_support)
dirac = np.zeros(len(delta))
for delta_loop in range(len(delta)):
    dirac[delta_loop] = delta[delta_loop]*(1/delta[delta_loop])*np.log(1/delta[delta_loop])

plt.plot(delta, dirac)