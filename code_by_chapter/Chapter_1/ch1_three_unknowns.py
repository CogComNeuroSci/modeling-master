"""
plot a function of two variables in two different ways
Code by Pieter Huycke
"""

from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, show
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

# the function that I'm going to plot
def z_func(x, y):
    #return (1 - (x ** 2 + y ** 3)) * exp(-(x ** 2 + y ** 2) / 2)
	#return x**2 + y**2
	return (x-y)**2


x = arange(-3.0, 3.0, 0.1)
y = arange(-3.0, 3.0, 0.1)

# grid of point
X, Y = meshgrid(x, y)

# evaluation of the function on the grid
Z = z_func(X, Y)

fig = plt.figure()


ax1 = fig.add_subplot(1, 2, 1)
# a plot from above 
im = ax1.imshow(Z, cmap = cm.RdBu)
# adding the contour lines with labels
cset = ax1.contour(Z, arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
ax1.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

# adding the colobar on the right
fig.colorbar(im, ax = ax1)

# and the 3D plot
ax2 = fig.add_subplot(1, 2, 2, projection = "3d")
surf = ax2.plot_surface(X, Y, Z, rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

