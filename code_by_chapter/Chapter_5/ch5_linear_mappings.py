#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 10:52:51 2020

@author: tom verguts
several linear cuts in 2D space
convenient to make slides illustrating how backprop works
"""

import matplotlib.pyplot as plt
import numpy as np

w = np.array([[1, 1, 0], [2, -1, 2], [-1.5, 2, 3], [-1, -1, -1]])

fig, axs = plt.subplots(nrows = 2, ncols = 2)

x = np.linspace(-1, 1)

for loop in range(4):
	row = loop//2
	col = loop%2
	y = w[loop, 0]*x + w[loop, 1]
	axs[row, col].plot(x, y, color = "black")
	axs[row, col].set_xlim(left = -1, right = 1)
	axs[row, col].set_ylim(bottom = -1, top = 1)