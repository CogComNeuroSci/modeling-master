#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:54:33 2020

@author: tom verguts
to illustrate vanishing and exploding gradients
see MCP book p 76
"""

import numpy as np
import matplotlib.pyplot as plt

starting_point = 1
x = np.linspace(0, 10)
r = 1.8 # the reproduction factor; r < 1 is vanishing, r > 1 is exploding

y = starting_point*np.power(r, x)

fig, ax = plt.subplots(1, 1)

ax.plot(x, y, color = "black")
ax.set_xlim(min(x), max(x))
ax.set_ylim(0, 10)