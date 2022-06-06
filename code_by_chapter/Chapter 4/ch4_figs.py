#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom verguts
create plots from chapter 4
"""

#%% imports and initialisations
import numpy as np
import matplotlib.pyplot as plt

def logistic(x,y,slope,intercept):
    return 1/(1+np.exp(slope*x+intercept-y))

low, high = -2, 4
font_size = 12
x = np.linspace(low,high,20)

#%% fig 4.2
plt.figure(0)
plt.suptitle("activation functions")

# linear activation function
y = x + 1
plt.subplot(321) # 3*2 plot; leave the middle two empty for breathing
plt.plot(x,y,color="black")
plt.title("a)", {"fontsize": font_size})
plt.xlabel("x", fontsize = font_size)
plt.ylabel("y", fontsize = font_size)

# threshold activation function
threshold = 1
y = (x>threshold)
plt.subplot(322)
plt.plot(x,y,color="black")
plt.title("b)", {"fontsize": font_size})
plt.xlabel("x", fontsize = font_size)
plt.ylabel("y", fontsize = font_size)

# sigmoid activation function
# the same threshold is used in a soft way in this case
y = 1/(1+np.exp(-2*(x-threshold)))
plt.subplot(325)
plt.plot(x,y,color="black")
plt.title("c)", {"fontsize": font_size})
plt.xlabel("x", fontsize = font_size)
plt.ylabel("y", fontsize = font_size)

# Gaussian activation function
y = np.exp(-(x-1)**2)
plt.subplot(326)
plt.plot(x,y,color="black")
plt.title("d)", {"fontsize": font_size})
plt.xlabel("x", fontsize = font_size)
plt.ylabel("y", fontsize = font_size)

#%% fig 4.3
# geometric intuition for threshold function
plt.figure(1)
plt.suptitle("geometric intuition for threshold function")
plt.subplot(121)
#plt.title("fig 4.3a: \nthreshold activation function in 2D", {"fontsize": font_size})
slope, intercept = 2, 1
y = slope*x + intercept
plt.plot(x,y,color="black")
n_dots = 15 # ndots randomly chosen points in space
color = "k"
for _ in range(n_dots):
    dot = ([np.random.uniform(low,high), 
            np.random.uniform(slope*low+intercept,slope*high+intercept)])
    if slope*dot[0] + intercept > dot[1]:
        dot_type = "o"
    else:
        dot_type = "+"
    plt.plot(dot[0],dot[1],color+dot_type)
    
# geometric intuition for sigmoid function
plt.subplot(122)
#plt.title("fig 4.3b: \nlogistic activation function in 2D", {"fontsize": font_size})
n_grid = 100
xi = np.linspace(low, high, n_grid)
yi = np.linspace(slope*low + intercept, slope*high + intercept, n_grid)
Xi, Yi = np.meshgrid(xi, yi)
zi = logistic(Xi,Yi,slope,intercept)
plt.contourf(xi, yi, zi, 14, cmap = "gray")

#%% representing the threshold function (fig 4.4)
plt.figure(2)
plt.suptitle("the threshold function (fig 4.4)")
plt.subplot(131)
#plt.title("fig 4.4a: \nthreshold in a 1-dimensional function", fontsize = 9)
plt.xlabel("x")
plt.ylabel("y")
plt.text(0.1, 0.2, "}", fontsize = 20)
plt.text(0.8, 0.3, r"$\theta$")
xi = np.linspace(low, high, n_grid)
yi = np.linspace(slope*low + intercept, slope*high + intercept, n_grid)
plt.plot(xi, yi, color = "black")
plt.plot(xi,[0]*len(xi),  color = "black")
plt.plot([0]*len(xi), yi, color = "black")
w1, w2, threshold = -1, 2, 1

plt.subplot(133)
#plt.title("fig 4.4b: \nthreshold in a 2-dimensional function", fontsize = 9)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.text(1.1, -0.6, "}", fontsize = 12)
plt.text(1.8, -0.6, r"$|\theta/w_2|$")
plt.text(-0.2, -1.5, "}", fontsize = 20, rotation = -90)
plt.text(0.2, -2.2, r"$|\theta/w_1|$")
plt.plot(xi,[0]*len(xi),  color = "black")
plt.plot([0]*len(xi), yi, color = "black")
yi = np.linspace(-w1/w2*low-threshold/w2, -w1/w2*high-threshold/w2, n_grid)
plt.plot(xi, yi, color = "black")

#%% linearly and non-linearly separable tasks (fig 4.6)
plt.figure(3)
plt.suptitle("logical rules")
fontsize = 15
left, right, down, up = 0, 1, 0, 1
mapping = [[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]]
text = ["fig 4.6a:\nThe OR problem", "fig 4.6b:\nThe AND problem", "fig 4.6c:\nThe XOR problem"]
for loop in range(3):
    plt.subplot(3,3,loop+1)
    plt.title(text[loop], fontsize = 10)
    plt.axis("off")
    plt.subplot(3,3,3+loop+1)
    plt.xticks(x)
    plt.yticks(y)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.text(left, down, mapping[loop][0], fontsize = fontsize)
    plt.text(left, up,   mapping[loop][1], fontsize = fontsize)
    plt.text(right, down,mapping[loop][2], fontsize = fontsize)
    plt.text(right, up,  mapping[loop][3], fontsize = fontsize)


#%% linearly (in)dependent vectors
plt.figure(4)
plt.suptitle("linearly (in)dependent vectors")
stretch = 1.1
v1 = np.array([1, -1, 1*stretch, 0]) # vector 2 is a stretched version of vector 0 (with a stretch factor stretch)
v2 = np.array([1,  1, 1*stretch, 1]) # vector 2 is a stretched version of vector 0 (with a stretch factor stretch)
x_locations = y_locations = [0, 0, 0, 0]
plt.quiver(x_locations, y_locations, v1, v2, angles='xy', scale_units='xy', scale=1)
plt.xlim(-1.5, 1.5)

plt.xlabel("Input dimension 1")
plt.ylabel("Input dimension 2")
plt.ylim(-1.5, 1.5)
plt.show()