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
s2_prior = 1 # also called parameter ksi when varied
s2_noise = 1
theta_prior = 0.5
theta_value = 1.0
x = 1 # a data point!
n_support = 50
v_d = np.linspace(low_d, high_d, num = n_support)
v_t = np.linspace(low_t, high_t, num = n_support) # vector of points where density is calculated
v_k = np.linspace(low_k, high_k, num = n_support)
delta_d = v_k[1]-v_k[0]
delta_t = v_t[1]-v_t[0]

def g(input):
    return input

def jointprob(data, th, ksi): ## in log scale
    return 1/2*( -np.log(ksi) - ((th-theta_prior)**2)/ksi - np.log(s2_noise) \
                 - ( (data-g(th))**2 )/s2_noise ) -np.log(2*np.pi)

def condprob(data, th):       ## in log scale
    return 1/2*(-np.log(s2_noise) - ((data - g(th))**2)/s2_noise -np.log(2*np.pi))

def q(v):
    return 1/np.sqrt(2*np.pi)*np.exp(-1/2*(v**2))

fig, axes = plt.subplots(nrows = 1, ncols = 2)

logprob  = np.zeros(len(v_t))
for theta_loop in range(len(v_t)):
    logprob[theta_loop] = jointprob(x, v_t[theta_loop], s2_prior)
      
axes[0].plot(v_t, logprob, color = "black")      

logprob = np.zeros(len(v_k))
F = np.zeros(len(v_k))
surprise = np.zeros(len(v_k))

for ksi_loop in range(len(v_k)):
#    logprob[ksi_loop] = jointprob(x, theta_value, v_k[ksi_loop])
    for theta_loop in range(len(v_t)): # F is calculated using a proper function q()
        F[ksi_loop] += \
         q(v_t[theta_loop]-theta_value)*np.log(np.exp(jointprob(x, v_t[theta_loop], v_k[ksi_loop]))/q(v_t[theta_loop]-theta_value))
    
#    for data_loop in range(len(v_d)):
#        for theta_loop in range(len(v_t)):
#            surprise[ksi_loop] += \
#                np.exp(jointprob(v_d[data_loop],v_t[theta_loop],v_k[ksi_loop]))*\
#                condprob(v_d[data_loop],v_t[theta_loop])
    for theta_loop in range(len(v_t)):
        surprise[ksi_loop] += \
                np.exp(jointprob(x,v_t[theta_loop],v_k[ksi_loop]))
#    surprise[ksi_loop] += \
#                np.exp(jointprob(x,theta_value,v_k[ksi_loop]))           
         
surprise = np.log(surprise*delta_t)
F = F*delta_t

#axes[1].plot(v_k, logprob, color = "black")    
axes[1].plot(v_k, surprise, color = "red")
axes[1].plot(v_k, F, color = "green")