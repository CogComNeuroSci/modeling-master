#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:42:02 2019

@author: tom verguts
chapter 12: interacting organisms
Specifically, interacting fireflies with the Kuramoto model (for overview,
see Strogatz, 2000, Physica D)
Illustrates gradual convergence of luminance in fireflies
also called flashing in synchrony: actual wave is not sinusoid though, this is an approximation
The fireflies have disappeared from the chapter, but code could stil be of interest
"""

#%% import and initialization
import numpy as np
import matplotlib.pyplot as plt

K = 0.3 # coupling strenght between the oscillators
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

#%% a: sine waves
for loop in range(len(w)):
    theta[loop, :] = w[loop]*points + np.random.rand()*(2*np.pi)
    c[loop, :] = np.sin(theta[loop,:]+phase[loop])
    axes[0].plot(points, c[loop, :], color = colors[loop])
axes[0].set_ylabel("luminance")
axes[0].set_title("sine waves")

#%% b: converging oscillators in Kuramoto model
theta = np.zeros((len(w), n_elem)) # location
theta[:,0] = np.random.rand(len(w))*(2*np.pi) # initial location

for time_loop in steps:
    for loop in range(len(w)):
        total = 0
        for j_loop in range(len(w)): # push-pull with all the other "fireflies"
            total += np.sin(theta[j_loop, time_loop-1]-theta[loop, time_loop-1])
        total *= K/len(w)    
        theta[loop, time_loop] = theta[loop, time_loop-1] + delta*(total + w[loop]) # eq (3.1) in Strogatz
        c[loop, time_loop] = np.sin(theta[loop, time_loop]+phase[loop])      

for loop in range(len(w)):
    axes[1].plot(points, c[loop, :], color = colors[loop])
axes[1].set_ylabel("luminance")
axes[1].set_xlabel("time")
axes[1].set_title("converging oscillators")