#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:37:54 2019

@author: tom
implements free energy (aka - joint probability)
"""
#%% initialize
import numpy as np
import matplotlib.pyplot as plt

low_d, high_d = -1, +1
low_t, high_t = -1, +1
low_k, high_k =  0.1,  1
s2_prior = 1
s2_noise = 0.3
theta_prior = 0
theta_value = 0.4
x = 0.3 # a data point!
n_support = 50
v_d = np.linspace(low_d, high_d, num = n_support)
v_t = np.linspace(low_t, high_t, num = n_support) # vector of points where density is calculated
v_k = np.linspace(low_k, high_k, num = n_support)
delta = v_k[1]-v_k[0]

def g(input):
    return input

def jointprob(data, th, ksi): ## in log scale
    return 1/2*( -np.log(ksi) - ((th-theta_prior)**2)/ksi - np.log(s2_noise) \
                 - ( (data-g(th))**2 )/s2_noise )

def condprob(data, th):       ## in log scale
    return 1/2*(-np.log(s2_prior) - ((data - g(th))**2)/s2_noise )

fig, axes = plt.subplots(nrows = 1, ncols = 2)

logprob  = np.zeros(len(v_t))
for theta_loop in range(len(v_t)):
    logprob[theta_loop] = jointprob(x, v_t[theta_loop], s2_prior)
      
axes[0].plot(v_t, logprob, color = "black")      

logprob = np.zeros(len(v_k))
surprise = np.zeros(len(v_t))
for ksi_loop in range(len(v_k)):
    logprob[ksi_loop] = jointprob(x, theta_value, v_k[ksi_loop])
#    for data_loop in range(len(v_d)):
#        for theta_loop in range(len(v_t)):
#            surprise[ksi_loop] += \
#                np.exp(jointprob(v_d[data_loop],v_t[theta_loop],v_k[ksi_loop]))*\
#                condprob(v_d[data_loop],v_t[theta_loop])

surprise = surprise*delta
    
axes[1].plot(v_k, logprob, color = "black")    
axes[1].plot(v_k, surprise, color = "red")