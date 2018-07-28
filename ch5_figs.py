#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:25:17 2018

@author: tom
pics from chapter 5
"""
import numpy as np
import matplotlib.pyplot as plt

slope = [ -2, 0.5, 3]
intercept = [0, 1, 1]
low, high = -1, 2
font_size = 5

def conv(x, y, sign = 1, list = range(len(slope))):
    inside = True
    for i in list:
        if sign*(slope[i]*x + intercept[i] - y) > 0:
            inside = False
    return int(inside)

def plotlines():
    x = np.linspace(low, high)
    for i in range(len(slope)):
        y = slope[i]*x + intercept[i]
        plt.plot(x, y, color = "black")
    
#%% figure 5.2a
# determine the convex set
plt.subplot(121)
plotlines()
ngrid = 100
xi = np.linspace(low, high, ngrid)
# clunky vector operation in python :-(
extremes = [np.add(np.multiply(low,slope),intercept), np.add(np.multiply(high,slope),intercept)]
yi = np.linspace(np.amin(extremes), np.amax(extremes), ngrid)
Xi, Yi = np.meshgrid(xi, yi)
Zi = np.empty(Xi.shape)
# can the convex function be vectorized?
for row in range(len(xi)):
    for column in range(len(yi)):
        Zi[row,column] = conv(Xi[row,column],Yi[row,column])
plt.contourf(xi, yi, Zi)
plt.title("Fig 5.2a \nAn AND of linear functions is a convex set", {"fontsize": font_size})

#%% figure 5.2b
plt.subplot(122)
plotlines()
Zi_full = np.zeros(Xi.shape)
checklist = [[0], [2]]
for index in checklist:
    for row in range(len(xi)):
        for column in range(len(yi)):
            Zi[row,column] = conv(Xi[row,column],Yi[row,column],sign = -1, list = index)
    Zi_full = np.any([Zi_full,Zi],axis=0)        
plt.contourf(xi, yi, Zi_full)
plt.title("Fig 5.2b \nAn OR of convex sets", {"fontsize": font_size})

#%% figure 5.5a
font_size = 10
plt.subplot(121)
x = np.arange(0,5)
y1 = (1/2)**x   # vanishing line, because base < 1
y2 = (1.5)**x   # exploding line, because base > 1
plt.plot(x,y1, color = "black")
plt.plot(x,y2, color = "black")
plt.xticks(x,5-np.arange(0,5))
plt.xlabel("Layer number (lower is deeper)")
plt.title("Fig 5.5a \nVanishing and exploding functions", {"fontsize": font_size})

#%% figure 5.5b
plt.subplot(122)
x = np.linspace(-2,2)
y = np.maximum(0,x) # rectified linear function
plt.plot(x,y, color = "black")
plt.xlabel("x")
plt.ylabel("y = max{0,x}")
plt.title("Fig 5.5b \nA novel transformation function", {"fontsize": font_size})
