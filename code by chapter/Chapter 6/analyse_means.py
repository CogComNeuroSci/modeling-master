#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:57:58 2020

@author: tom
"""

import numpy as np


v = np.load("simulation_results_50_Powellbayes.npy")
print(np.mean(v, axis=0))
print(np.std(v, axis=0))