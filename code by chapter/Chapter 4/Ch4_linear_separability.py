#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:09:37 2020

@author: tom
illustration that the task-switch stroop task is nonlin separable
"""

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_aspect("equal")
r = [-1, 1]
product_set = list(product(r, r, r))
list1= [(1, 1, 1),  (-1, 1, 1),  (1, 1, -1), (1, -1, -1)]
list2= [(1, -1, 1), (-1, -1, 1), (-1, 1, -1), (-1, -1, -1)]
color_list = ["black", "red"]

# draw cube
for s, e in combinations(np.array(product_set), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="black", linewidth = 1)
for (idx, list_loop) in enumerate([list1, list2]):	
	col = color_list[idx]
	for loop in list_loop:		
		ax.scatter3D(loop[0], loop[1], loop[2], c = col, s = 100);
