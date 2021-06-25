#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:15:46 2017

@author: tom
plots chapter 2, decision making
"""

# import
import numpy as np
from numpy import maximum
from numpy import random
import matplotlib.pyplot as plt

#figures 2.1b and 2.1c
#simple energy minimization model
#with RT simulation

# initialize
n_trials = 20000
b = [3, 0.2]
w_p = 0.01
w_m = 1
theta = 0.995 # 1-decay parameter
alpha = 0.01 # change rate
max_n_step = 300
std_noise = 0.1
threshold=2
rt = []
accuracy = 0
too_late = 0

for loop in range(n_trials):
    # start
    x=[0]
    y=[0]
    threshold_reached = False
    counter = 0
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
    rt.append(counter)
    accuracy += (x[-1]>y[-1])    
    too_late += (counter>=max_n_step)

# plot final trajectory
f, axarr = plt.subplots(nrows = 2, ncols = 1)
axarr[0].plot(range(len(x)),x,"ko-", range(len(y)),y,"k--")
axarr[0].set_title("Random trajectory")
#axarr[0].set_xlabel("Time")
axarr[0].set_ylabel("Activation")
# plot histogram
axarr[1].hist(rt, bins = 50, color = "black")
axarr[1].set_title("RT distribution")
#axarr[0].set_xlabel("Time")
axarr[1].set_ylabel("Frequency")
print("accuracy: {:2.0%}".format(accuracy/n_trials))
print("too late: {:2.0%}".format(too_late/n_trials))

#plots fig 2.5
#
fig = plt.subplots()
x = np.linspace(start = -4, stop = 3, num = 100)
y = (x**4)/4 +(2./3)*(x**3)-(5./2)*(x**2)-6*x
#y = x**3 + 2*(x**2) - 5*x - 6
plt.plot(x,y, color = "black")
plt.ylabel("y = f(x)")
