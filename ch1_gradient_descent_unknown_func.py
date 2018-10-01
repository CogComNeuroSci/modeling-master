#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:34:00 2018

@author: mehdisenoussi
"""

import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import cm
import pylab as pl
import numpy as np
import time
# from psychopy


def quad_func(x):
    return 3 * x**2 - 2 * x + 4

def other_func1(x):
    return 5/(.7*x**3 + 10*np.sin(x)) + .5*x**2

def other_func2(x):
    return 5/(.7*x**3 + 10*np.sin(x)) + 10*np.cos(x) + .5*x**2

def other_func3(x):
    return 5*x/(7*x**3 - 10*np.sin(x)) + 10*np.cos(x) + .5*x**2

n_steps = 20

norm = mpl.colors.Normalize(vmin = 0, vmax = 1)

func_to_use = quad_func#other_func2

x = np.linspace(-20, 20, 5000)
y = func_to_use(x)

fig, axes = pl.subplots(1, 1)
axes.plot(x, y, 'k')

x_desc = []
x_desc.append(np.random.choice(x))
alpha = .2
y_desc = []
y_desc.append(func_to_use(x_desc[-1]))
col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba(0)
axes.plot(x_desc[-1], y_desc[-1], 'o', mec = 'k', color = col)

# random first step
x_desc.append(x_desc[0] * np.random.random()*2)
y_desc.append(func_to_use(x_desc[-1]))
col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba(1/float(n_steps+2))
axes.plot(x_desc[-1], y_desc[-1], 'o', mec = 'k', color = col)

for i in np.arange(n_steps):
    pl.title('step: %i/%i' % (i+2, n_steps+1))
    der = (y_desc[-1] - y_desc[-2]) / (x_desc[-1] - x_desc[-2])
    x_desc.append(x_desc[-1] + -alpha * der)
    y_desc.append(func_to_use(x_desc[-1]))
    col = cm.ScalarMappable(norm = norm, cmap = cm.afmhot).to_rgba((i+2)/float(n_steps+2))
    a = axes.plot(x_desc[-1], y_desc[-1], 'o', mec = 'k', color = col)
    axes.set_xlabel('alph * der = x_step\n%.2f * %.2f = %.2f' % (alpha, der, der*alpha))
    fig.canvas.draw()
    fig.waitforbuttonpress(0)
    


pl.title('THE END\nstep: %i/%i' % (i+2, n_steps+1))




