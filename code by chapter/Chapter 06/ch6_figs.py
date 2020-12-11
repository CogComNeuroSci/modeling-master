#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2, 2018

@author: tom verguts
pics and tables from chapter 6
"""

import numpy as np
import matplotlib.pyplot as plt

#%% fig 6.2: taking logs doesn't change optimum
x = np.linspace(-3, 3, 50)
y1 = x**2
y2 = np.log(y1)

fig, axs = plt.subplots(nrows = 1, ncols = 2)

axs[0].plot(x, y1, color = "black")
axs[0].set_title("$y = x^2$")
axs[1].plot(x, y2, color = "black")
axs[1].set_title("$y = log(x^2)$")

#%% fig 6.3 log likelihood: dependence on data size
logL_range = 100
n = [[7, 3], [70, 30]]
fig, axs = plt.subplots(nrows = 1, ncols = 2)

p = np.linspace(0+1/50, 1-1/50, 50)
for loop in range(2):
    logL = n[loop][0]*np.log(p) + n[loop][1]*np.log(1-p)
    av = np.mean(logL)
    axs[loop].plot(p, logL, color = "black")
    axs[loop].set_ylim(av-logL_range/2, av+logL_range/2)
    axs[loop].set_title("{} data points".format(n[loop][0]+n[loop][1]))

#%% fig 6.4: grid
x = np.linspace(0, 1, 10)
y = np.linspace(0, 5, 10)

plt.figure()
plt.xticks(x)
plt.yticks(y)
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.grid(True)

#%% table: illustration of the softmax function
w = np.array([2, 0.2, 0.5, 1])
gamma = [0.2, 2] # low and high value
for gamma_loop in gamma:
    denominator = np.sum(np.exp(gamma_loop*w))
    print( np.exp(gamma_loop*w)/denominator )
    
#%% table: illustration of AIC, BIC, and transfer test
"""
AIC and BIC illustration
model 1: There is just one p-value
model 2: Part 1 and part 2 have a different p-value
"""
n_trials = 1000
p = [[0.6, 0.6], [0.5, 0.7]]
K = [1, 2] # nr of parameters

for loop in range(2):
    data = np.zeros(n_trials)
    data_new = np.zeros(n_trials) # these are the transfer data
    for data_loop in range(n_trials):
        data[data_loop] = (np.random.rand() > p[loop][data_loop//(n_trials//2)])
        data_new[data_loop] = (np.random.rand() > p[loop][data_loop//(n_trials//2)])
    n_heads = np.array([np.sum(data[:n_trials//2+1]), np.sum(data[n_trials//2:])])
    n_heads_new = np.array([np.sum(data_new[:n_trials//2+1]), np.sum(data_new[n_trials//2:])])
    p1 = np.sum(n_heads)/data.size # p-values estimates only on data (not data_new)
    p2 = np.array(n_heads/(data.size//2))

    lik1 = np.sum(n_heads)*np.log(p1) + (n_trials-np.sum(n_heads))*np.log(1-p1)
    lik2 = ( n_heads[0]*np.log(p2[0]) + (n_trials//2-n_heads[0])*np.log(1-p2[0])+
            n_heads[1]*np.log(p2[1]) + (n_trials//2-n_heads[1])*np.log(1-p2[1]) )

    aic1 = -2*lik1 + 2*K[0]
    aic2 = -2*lik2 + 2*K[1]

    bic1 = -2*lik1 + K[0]*np.log(n_trials)/2
    bic2 = -2*lik2 + K[1]*np.log(n_trials)/2

    lik1_new = np.sum(n_heads_new)*np.log(p1) + (n_trials-np.sum(n_heads_new))*np.log(1-p1)
    lik2_new = ( n_heads_new[0]*np.log(p2[0]) + (n_trials//2-n_heads_new[0])*np.log(1-p2[0])+
            n_heads_new[1]*np.log(p2[1]) + (n_trials//2-n_heads_new[1])*np.log(1-p2[1]) )
    print("Dataset {0}\n LogL model1: {1:.2f} model 2: {2:.2f}\n AIC model1: {3:.2f} \
           model2: {4:.2f}\n BIC model1 {5:.2f} BIC model2 {6:.2f} LogLtransf model1: {7:.2f} LogLtransf model2: {8:.2f}".
          format(loop+1,lik1, lik2, aic1, aic2, bic1, bic2, lik1_new, lik2_new))