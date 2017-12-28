#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:15:46 2017

@author: tom
simple energy minimization model
"""

# import
from numpy import maximum
from numpy import random
import matplotlib.pyplot as plt

# initialize
b = [3, 2]
w_p = 0.1
w_m = 1
theta = 0.99 # decay parameter
alpha = 0.01 # change rate
max_n_step = 1000
x=[0]
y=[0]
std_noise = 0.1
threshold=2
threshold_reached = False
counter = 0

# start
while not threshold_reached and counter<max_n_step:
    d_x = alpha*(b[0] + 2*w_p*x[-1] - w_m*y[-1]) + std_noise*random.randn()
    d_y = alpha*(b[1] + 2*w_p*y[-1] - w_m*x[-1]) + std_noise*random.randn()
    x.append(theta*x[-1] + d_x)
    y.append(theta*y[-1] + d_y)
    x[-1] = maximum(0,x[-1]) # threshold at zero
    y[-1] = maximum(0,y[-1]) # threshold at zero
    if maximum(x[-1],y[-1])>=threshold:
        threshold_reached = True
    counter += 1    
plt.plot(range(len(x)),x,"bo-", range(len(y)),y,"ro-")
##