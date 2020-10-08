#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:37:54 2019

@author: tom verguts
implements free energy (aka - joint probability)
"""
#%% initialize
import numpy as np
import matplotlib.pyplot as plt

low_theta, high_theta = -10, +10
low_ksi, high_ksi =  0.1,  2
s2_prior = 0.1 
s2_noise = 1
theta_prior = 0.5 # also called parameter ksi when varied
theta_value = 1
x = 1 # a data point!
n_support = 100
v_theta = np.linspace(low_theta, high_theta, num = n_support) # vector of points where density is calculated
v_ksi = np.linspace(low_ksi, high_ksi, num = n_support)
delta_theta = v_theta[1]-v_theta[0]

def g(input):
    return input

def jointprob(data, theta, ksi): ## in log scale
    return 1/2*( -np.log(ksi) - ((theta-theta_prior)**2)/ksi - np.log(s2_noise) \
                 - ( (data-g(theta))**2 )/s2_noise ) -np.log(2*np.pi)

def condprob(data, theta):       ## in log scale
    return 1/2*(-np.log(s2_noise) - ((data - g(theta))**2)/s2_noise -np.log(2*np.pi))

def thetaprob(theta, ksi):      ## in log scale
    return 1/2*(-np.log(ksi) -       ((theta-theta_prior)**2)/ksi - np.log(2*np.pi))

#%% part 1
fig, axes = plt.subplots(nrows = 1, ncols = 2)

logprob  = np.zeros(len(v_theta))
for theta_loop in range(len(v_theta)):
    logprob[theta_loop] = jointprob(x, v_theta[theta_loop], s2_prior)
axes[0].plot(v_theta, logprob, color = "black")      
axes[0].set_xlabel("\u03B8")
axes[0].set_ylabel("log f(\u03B8,X;\u03BE)")
axes[0].set_title("Perception")

#%% part 2
F = np.zeros(len(v_ksi))
for ksi_loop in range(len(v_ksi)):
    F[ksi_loop] = jointprob(x, theta_value, v_ksi[ksi_loop])             
     
axes[1].set_xlabel("\u03BE")
axes[1].set_title("Learning")
axes[1].plot(v_ksi, F, color = "black")