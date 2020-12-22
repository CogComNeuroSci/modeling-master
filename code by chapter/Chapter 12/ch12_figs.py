#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:42:02 2019

@author: tom verguts
chapter 12: interacting organisms
Specifically, interacting fireflies with the Kuramoto model
The fireflies have disappeared from the chapter, but could stil be of interest
"""

import numpy as np
import matplotlib.pyplot as plt

## figure 12.1
K = 0.3
n_elem = 1000
delta = 0.05
colors = ["black", "blue", "red", "green"]

w =     [1, 1.05, 1.1, 1.2] # frequencies
phase = np.zeros(len(w))  # zero phases
#phase = np.random.randn(len(w))  # random phases
theta = np.zeros((len(w), n_elem)) # location
c     = np.zeros((len(w), n_elem)) # circle
steps = np.array(range(n_elem-1)) + 1
points = np.linspace(0, delta*n_elem, num = n_elem)
fig, axes = plt.subplots(nrows = 2, ncols = 1)

## figure 12.1a
for loop in range(len(w)):
    theta[loop, :] = w[loop]*points + np.random.rand()*(2*np.pi)
    c[loop, :] = np.sin(theta[loop,:]+phase[loop])
    axes[0].plot(points, c[loop, :], color = colors[loop])
axes[0].set_ylabel("luminance")
axes[0].set_title("12.1a")

## figure 12.1b
theta = np.zeros((len(w), n_elem)) # location
theta[:,0] = np.random.rand(len(w))*(2*np.pi) # initial location

for time_loop in steps:
    for loop in range(len(w)):
        sum = 0
        for j_loop in range(len(w)):
            sum += np.sin(theta[j_loop, time_loop-1]-theta[loop, time_loop-1])
        sum *= K/len(w)    
        theta[loop, time_loop] = theta[loop, time_loop-1] + delta*(sum+w[loop])
        c[loop, time_loop] = np.sin(theta[loop, time_loop]+phase[loop])      

for loop in range(len(w)):
    axes[1].plot(points, c[loop, :], color = colors[loop])
axes[1].set_ylabel("luminance")
axes[1].set_xlabel("time")
axes[1].set_title("12.1b")