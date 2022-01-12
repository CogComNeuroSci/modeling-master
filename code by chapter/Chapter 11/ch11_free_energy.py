#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:37:54 2019

@author: tom verguts
implements free energy (aka - log joint probability of data and parameter theta)
there is no q(theta) function here because distributions of theta and ksi reduce to a point distribution (as in Bogacz, JMP, 2015)
"""
#%% initialize
import numpy as np
import matplotlib.pyplot as plt

low_theta, high_theta = -10, +10 # for plotting fig 11.3b (perception)
low_ksi, high_ksi =  0.1,  2     # for plotting fig 11.3b (learning)
s2_noise = 1      # data variance
s2_prior = 0.1    # fixed value of parameter prior variance (also called ksi when varied)
theta_prior = 0.5 # parameter prior mean
theta_value = 1
x = 1 # a data point!
n_support = 100
v_theta = np.linspace(low_theta, high_theta, num = n_support) # vector of points where density is calculated
v_ksi = np.linspace(low_ksi, high_ksi, num = n_support)
delta_theta = v_theta[1]-v_theta[0]

def g(parameter):
    """data can in general be a function g() of the parameter theta; see Bogacz, JMP, 2015"""
    return parameter

def jointprob(data, theta, ksi):
    """joint probability of data and theta in log scale; decomposition is as in Bogacz, JMP, 2015, eq (7)"""
    return (1/2*( -np.log(ksi) -np.log(s2_noise) -((theta-theta_prior)**2)/ksi - ((data-g(theta))**2)/s2_noise )
           -np.log(2*np.pi))

def condprob(data, theta):       
    """conditional probability of data conditional on theta in log scale; not currently used"""
    return 1/2*(-np.log(s2_noise) - ((data - g(theta))**2)/s2_noise -np.log(2*np.pi))

def thetaprob(theta, ksi):
    """probability of theta, in log scale; not currently used"""
    return 1/2*(-np.log(ksi) -       ((theta-theta_prior)**2)/ksi - np.log(2*np.pi))

#%% part 1: maximizing theta gives you the best hypothesis (theta) about the state of the world, given the data X
fig, axes = plt.subplots(nrows = 1, ncols = 2)

logprob  = np.zeros(len(v_theta))
for idx, theta in enumerate(v_theta):
    logprob[idx] = jointprob(x, theta, s2_prior)
axes[0].plot(v_theta, logprob, color = "black")      
axes[0].set_xlabel("\u03B8")
axes[0].set_ylabel("log f(\u03B8,X;\u03BE)")
axes[0].set_title("Perception")

#%% part 2: maximizing ksi is learning; it gives you the best parameter estimate that generates states of the world
F = np.zeros(len(v_ksi))
for idx, ksi in v_ksi:
    F[idx] = jointprob(x, theta_value, ksi)             
     
axes[1].set_xlabel("\u03BE")
axes[1].set_title("Learning")
axes[1].plot(v_ksi, F, color = "black")