"""
Code by Pieter Huycke
"""
from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show


# the function that I'm going to plot
def z_func(x, y):
    return (1 - (x ** 2 + y ** 3)) * exp(-(x ** 2 + y ** 2) / 2)


x = arange(-3.0, 3.0, 0.1)
y = arange(-3.0, 3.0, 0.1)

# grid of point
X, Y = meshgrid(x, y)

# evaluation of the function on the grid
Z = z_func(X, Y)

# drawing the function
im = imshow(Z, cmap=cm.RdBu)

# adding the Contour lines with labels
cset = contour(Z, arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

# adding the colobar on the right
colorbar(im)

# latex fashion title
title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')

# show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
