#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom verguts
create figs (and table) of chapters 1
"""

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision = 3, suppress = True)

x = np.linspace(start = -1, stop = 3, num = 20)
xvals = [2.7, 2.02, 1.61, 1.37]
y = (x - 1)**2
plt.plot(x,y, color = "black")
plt.ylabel("$y = (x-1)^2$")
bottom, top = plt.ylim()
plt.scatter(xvals, [bottom]*4, color = "black", s = 80)
plt.scatter(xvals, (np.array(xvals)-1)**2, color = "black", s = 80)
plt.ylim((bottom, top))
# the table...

def y(x):       # the function we aim to optimize
    return (x-1)**2

def y_der(x):    # the derivative of the function we aim to optimize
    return 2*(x-1)    

n_steps = 100
x_start = 2.7 # random starting point
alpha = 2
data = np.zeros((n_steps,4))
data[0,0] = x_start
for step in range(n_steps):
    data[step,1] = y(data[step,0])
    data[step,2] = y_der(data[step,0])
    data[step,3] = -alpha*data[step,2]
    if step<n_steps-1:
        data[step+1,0] = data[step,0] + data[step,3]

print(data)