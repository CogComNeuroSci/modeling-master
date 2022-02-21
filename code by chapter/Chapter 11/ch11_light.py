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
import matplotlib.pyplot as plt

p_conc  = 0.5 # a priori probability of concavity
lik     = np.array([0.8, 0.2])   # prob x = 0 for z = 0, 1
z_label = ["concave", "convex"]  
x_label = ["shadow", "light"]

def logjoint(x, z, lik, priorz):
    """used for optimizing both z and lik"""
    return np.log(lik[z])*(1-x) + np.log(1-lik[z])*x + np.log(priorz)

def logjointn(x, n, z, lik, priorz):
    """used for optimizing both z and lik; if you have n data points"""
    return np.log(lik[z])*(n-x) + np.log(1-lik[z])*x + np.log(priorz)

def optimize_z(x, lik, priorz):
    return np.argmax(np.array([logjoint(x, 0, lik, priorz), logjoint(x, 1, lik, 1-priorz)]))

def optimize_lik(x, z):
    """optimize toward lik (first element only); maximum can be found explicitly but it's
	also based on the log joint function defined above"""
    return np.sum((1-x)*(1-z))/np.sum(1-z)

def free_energy(x, Q_p, prior):
	return Q_p*np.log(Q_p) + (1-Q_p)*np.log(1-Q_p) - Q_p*logjoint(x, 0, lik, prior) - (1-Q_p)*logjoint(x, 1, lik, 1-prior)

def free_energyn(x, n, Q_p, prior):
	"""FE with n data points"""
	return Q_p*np.log(Q_p) + (1-Q_p)*np.log(1-Q_p) - Q_p*logjointn(x, n, 0, lik, prior) - (1-Q_p)*logjointn(x, n, 1, lik, 1-prior)

def surprise(x, prior):
    return -np.log(np.exp(logjoint(x, 0, lik, prior)) + np.exp(logjoint(x, 1, lik, 1-prior)))

def surprisen(x, n, prior):
    return -np.log(np.exp(logjointn(x, n, 0, lik, prior)) + np.exp(logjointn(x, n, 1, lik, 1-prior)))
	   
# optimize toward Z
observation = "light"
x  = x_label.index(observation)
print("i think the truth is {}!".format(z_label[optimize_z(x, lik, p_conc)]))

# optimize toward parameter lik
z = np.array([z_label.index("concave")]*4) # the truth is always concave
x = [x_label.index("shadow"), x_label.index("light"), x_label.index("shadow"), x_label.index("shadow")] # observations are usually shady
x = np.array(x)
print("my estimate of prob(shadow/concave) equals {}".format(optimize_lik(x, z)))

# calculate free energy
Q_p = 0.5 # probability of concave in Q(Z)
observation = "shadow"
x  = x_label.index(observation)
print("free energy = {:.2f}".format(free_energy(x, Q_p, p_conc)))
print("surprise = {:.2f}".format(surprise(x, p_conc)))

# plot free energy and surprise
fig, ax = plt.subplots(1, 3)

# calculate free energy across different priors on Z; data are n data points
low_z, high_z, step_size = 0.1, 0.99, 0.001
z_vec = np.arange(low_z, high_z, step_size)
F = np.ndarray(z_vec.size)
s = np.ndarray(z_vec.size)
x, n = 3, 5
for idx, z in enumerate(z_vec):
    F[idx] = free_energyn(x, n, Q_p, z)
    s[idx] = surprisen(x, n, z)

ax[0].plot(z_vec, F, color = "black")
ax[0].plot(z_vec, s, color = "red")

# calculate free energy across different Q's; data are n data points
q_vec = np.arange(low_z, high_z, step_size)
F = np.ndarray(q_vec.size)
s = np.ndarray(q_vec.size)
prior_z = 0.5 # probability of concave in P(Z)
x, n = 0, 1 
for idx, q in enumerate(q_vec):
    F[idx] = free_energyn(x, 1, q, prior_z)
    s[idx] = surprisen(x, 1, prior_z)

ax[1].plot(z_vec, F, color = "black")
ax[1].plot(z_vec, s, color = "red")

# calculate free energy across different Q's; there are now 5 data points
x, n = 0, 5 # with more data
for idx, q in enumerate(q_vec):
     F[idx] = free_energyn(x, n, q, prior_z)
     s[idx] = surprisen(x, n, prior_z)

ax[2].plot(z_vec, F, color = "black")
ax[2].plot(z_vec, s, color = "red")
