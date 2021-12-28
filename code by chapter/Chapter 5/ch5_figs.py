#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:25:17 2018

@author: tom verguts
pics from chapter 5
"""

#%% imports and initializations
import numpy as np
import matplotlib.pyplot as plt

slope = [ -2, 0.5, 3]
intercept = [0, 1, 1]
low, high = -1, 2
low_rbf, high_rbf = -3, 3
font_size = 10

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

def rbf(x,y,cen_x,cen_y):
    return np.exp(-((x-cen_x)**2 + (y-cen_y)**2))
    
#%% outputs of 2- and 3-layer networks 
# figure 5.3a
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
        Zi[row,column] = 2*conv(Xi[row,column],Yi[row,column])-1
plt.contourf(xi, yi, Zi, extend = "both", colors = "none", hatches=['', '.'])
plt.title("Fig 5.3a \nAn AND of linear functions (dots) \n is a convex set", {"fontsize": font_size})

# figure 5.3b
plt.subplot(122)
plotlines()
Zi_full = np.zeros(Xi.shape)
checklist = [[0], [1], [2]]
for index in checklist:
    for row in range(len(xi)):
        for column in range(len(yi)):
            Zi[row,column] = conv(Xi[row,column],Yi[row,column],sign = -1, list = index)
    Zi_full = np.any([Zi_full,Zi],axis=0)   
Zi_full = 2*Zi_full-1
plt.contourf(xi, yi, Zi_full, extend = "both", colors = "none", hatches=['', '.'])
plt.title("Fig 5.3b \nAn OR of convex sets (dots)", {"fontsize": font_size})

#%% figure 5.4: Local and global minima
# (Code mostly stolen from Pieter Huycke)

from numpy import arange
#from mpl_toolkits.mplot3d import Axes3D
#from pylab import meshgrid, cm, title
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

texts = ["5.3a:\nFunction with one (global) minimum", 
         "5.3b:\nFunction with several local minima"]

#texts = ["(a)", "(b)"]

# the function that I'm going to plot
def z_func(x, y, nr):
    if nr == 1:
        return (x ** 2 + y ** 2)/10
    else:
        return (x/4) ** 4 - (x/2) ** 2 + (y/4) ** 4 - (y/2) ** 2  + x/8 + y/8

x = arange(-8.0, 8.0, 0.1)
y = arange(-8.0, 8.0, 0.1)

# grid of point
X, Y = np.meshgrid(x, y)

fig = plt.figure()

for loop in range(2):
    Z = z_func(X, Y, loop+1)
    ax = fig.add_subplot(1, 2, loop+1, projection = "3d")
    ax.set_title(texts[loop])
#    ax.plot_surface(X, Y, Z, cmap="Greys")
    ax.plot_surface(X, Y, Z, cmap = plt.cm.coolwarm)
    # drawing the function
    ax.zaxis.set_major_locator(LinearLocator(4))
    ax.xaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#%% This figure did not survive in the final version of the MCP book
# vanishing and exploding gradients
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

# RELU transformation function
plt.subplot(122)
x = np.linspace(-2,2)
y = np.maximum(0,x) # rectified linear function
plt.plot(x,y, color = "black")
plt.xlabel("x")
plt.ylabel("y = max{0,x}")
plt.title("Fig 5.5b \nA novel transformation function", {"fontsize": font_size})


#%% figure 5.7 Radial basis functions
x_center, y_center = 1, 0
plt.figure()
plt.title("fig 5.7: \nRadial basis activation function", {"fontsize": font_size})
ngrid = 100
xi = np.linspace(low_rbf, high_rbf, ngrid)
yi = np.linspace(low_rbf, high_rbf, ngrid)
Xi, Yi = np.meshgrid(xi, yi)
zi = rbf(Xi,Yi,x_center,y_center)
plt.contourf(xi, yi, zi, 14, cmap="Greys")