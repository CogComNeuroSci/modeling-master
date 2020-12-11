#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:34:30 2018

@author: tom verguts
generate data for test 2 (MCP, Dec 2018)
"""
import numpy as np
import matplotlib.pyplot as plt

file_name = "test2_data"
n_stimuli = 40
n_dim = 2
std = 3
radius2 = 8
w = [1, 2] # for linear separation
data = np.random.randn(n_stimuli, n_dim)*std
lin_cat = np.dot(data, w) > 0
nonlin_cat = np.diag( np.dot(data, np.transpose(data)) ) > radius2
data = np.column_stack((data, lin_cat, nonlin_cat))
lin_color = np.array(["red", "green"])[[lin_cat*1]]
nonlin_color = np.array(["red", "green"])[[nonlin_cat*1]]
plt.scatter(data[:,0], data[:,1], c = nonlin_color)

np.save(file_name, data)