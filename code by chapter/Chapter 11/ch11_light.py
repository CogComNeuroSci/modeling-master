#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:44:15 2022

@author: tom verguts
bayes optimization of the example described by Smith et al, 2021, PsychRxiv
illustrates how the log joint density can be used to optimize both Z and the parameters
see MCP book, figure 11.3b and accompanying text
"""
import numpy as np

p_conc  = 0.45
prior   = np.array([p_conc, 1-p_conc]) # z = 0, 1; should sum to 1
lik     = np.array([0.9, 0.1])   # x = 0, 1 for z = 0, 1
z_label = ["concave", "convex"]  
x_label = ["shadow", "light"]

def logjoint(x, z, lik):
    """used for optimizing both z aand lik"""
    return np.log(lik[z])*(1-x) + np.log(1-lik[z])*x + np.log(prior[z])

def optimize_z(x, lik):
    return np.argmax(np.array([logjoint(x, 0, lik), logjoint(x, 1, lik)]))

def optimize_lik(x, z):
    """optimize toward lik (first element only); maximum can be found explicitly but it's
	also based on the log joint function defined above"""
    return np.sum((1-x)*(1-z))/np.sum(1-z)

# optimize toward Z
observation = "light"
x  = x_label.index(observation)
print("i think the truth is {}!".format(z_label[optimize_z(x, lik)]))

# optimize toward parameter lik
z = np.array([z_label.index("concave")]*4) # the truth is always concave
x = [x_label.index("shadow"), x_label.index("light"), x_label.index("shadow"), x_label.index("shadow")] # observations are usually shady
x = np.array(x)
print("my estimate of prob(shadow/concave) equals {}".format(optimize_lik(x, z)))