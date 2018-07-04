#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom
create plots from chapter 4
"""
import numpy as np
import matplotlib.pyplot as plt

def logistic(x,y,slope,intercept):
    return 1/(1+np.exp(slope*x+intercept-y))

low = -2
high = 4
font_size = 5
x = np.linspace(low,high,20)

# fig 4.2
plt.figure(0)
# linear activation function
y = x + 1
plt.subplot(321)
plt.plot(x,y,color="black")
plt.title("linear", {"fontsize": font_size})
# threshold activation function
threshold = 1
y = (x>threshold)
plt.subplot(322)
plt.plot(x,y,color="black")
plt.title("threshold", {"fontsize": font_size})
# sigmoid activation function
# the same threshold is used in a soft way in this case
y = 1/(1+np.exp(-2*(x-threshold)))
plt.subplot(325)
plt.plot(x,y,color="black")
plt.title("sigmoid (soft threshold)", {"fontsize": font_size})
# Gaussian activation function
y = np.exp(-(x-1)**2)
plt.subplot(326)
plt.plot(x,y,color="black")
plt.title("Gaussian", {"fontsize": font_size})

# fig 4.3
plt.figure(1)
# geometric intuition for threshold function
plt.subplot(121)
plt.title("fig 4.3a: \nthreshold activation function in 2D", {"fontsize": font_size})
slope = 2
intercept = 1
y = slope*x + intercept
plt.plot(x,y,color="black")
ndots = 15 # ndots randomly chosen points in space
for i in range(ndots):
    dot = ([np.random.uniform(low,high), 
            np.random.uniform(slope*low+intercept,slope*high+intercept)])
    if slope*dot[0] + intercept > dot[1]:
        color = "b"
    else:
        color = "y"
    plt.plot(dot[0],dot[1],color+"o")
    
# geometric intuition for sigmoid function
plt.subplot(122)
plt.title("fig 4.3b: \nlogistic activation function in 2D", {"fontsize": font_size})
ngrid = 100
xi = np.linspace(low, high, ngrid)
yi = np.linspace(slope*low + intercept, slope*high + intercept, ngrid)
Xi, Yi = np.meshgrid(xi, yi)
zi = logistic(Xi,Yi,slope,intercept)
plt.contourf(xi, yi, zi, 14)