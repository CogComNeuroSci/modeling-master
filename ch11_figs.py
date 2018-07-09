#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:37:20 2018

@author: tom
code for pics in chapter 11
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

# fig 11.2b
n_cats = 50
mean_cats = [-1, -1]
cov_cats  = np.eye(2)*.2
n_dogs = 50
mean_dogs = [1, 1]
cov_dogs  = np.eye(2)*.2
n_dogs_special = 10
mean_dogs_special = [0, 3]
cov_dogs_special  = np.eye(2)*.05
output1 = [-1.1, -0.9]
output2 = [0.9, 1.05]
cat_color = "black"
cat_marker = "o"
cat_size = 50
dog_color = "red"
dog_marker = "o"
dog_size = 50
cluster_color = "green"
cluster_marker = "X"
cluster_size= 200
cats = np.transpose(np.random.multivariate_normal(mean_cats,cov_cats,size = [n_cats]))
dogs = np.transpose(np.random.multivariate_normal(mean_dogs,cov_dogs,size = [n_dogs]))
dogs_special = np.transpose(np.random.multivariate_normal(mean_dogs_special,cov_dogs_special,size = [n_dogs_special]))
plt.scatter(cats[0],cats[1], n_cats, c = cat_color, marker = cat_marker)
plt.scatter(dogs[0],dogs[1], n_dogs, c = dog_color, marker = dog_marker)
plt.scatter(dogs_special[0], dogs_special[1], n_dogs, c = "orange")
plt.scatter(output1[0], output1[1], s = 200, c = cluster_color, marker = cluster_marker)
plt.scatter(output2[0], output2[1], s = 200, c = cluster_color, marker = cluster_marker)
plt.title("Competitive learning")

#%%
# fig 11.2c/d; embedded in a cell so it can run on its own
radius = 0.3
origin = [0.5, 0.5]
std = 0.2
n_dogs = 10
mean_angle_dogs = np.pi*(1/4)
n_cats = 10
mean_angle_cats = np.pi*(3/4)
fig, axes = plt.subplots(nrows = 1, ncols = 2)
circle = plt.Circle(origin, radius, edgecolor = "black", facecolor = "none")
angle_dogs    = np.random.normal(mean_angle_dogs, std,n_dogs)
angle_cats    = np.random.normal(mean_angle_cats, std,n_cats)
for index in range(2):
    circ = ptch.Circle(origin, radius, color = "black", fill = False)
    if index == 0:
        cluster_mean = np.random.uniform(0,2*np.pi,2)
    else:
        cluster_mean = [np.random.normal(mean_angle_dogs,std), np.random.normal(mean_angle_cats,std)]
    angle_cluster = cluster_mean
    for angle in [[angle_dogs, dog_color, dog_marker, dog_size], [angle_cats, cat_color, cat_marker, cat_size], [angle_cluster, cluster_color, cluster_marker, cluster_size]]:
        x = radius*np.cos(angle[0]) + origin[0]
        y = radius*np.sin(angle[0]) + origin[1]
        axes[index].scatter(x, y, c = angle[1], marker = angle[2], s = angle[3])
    axes[index].add_patch(circ)    
    axes[index].axis([0, 1, 0, 1])
#plt.show()    