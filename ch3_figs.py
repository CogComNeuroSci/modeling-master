#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom
create plots from chapter 3
"""
import matplotlib.pyplot as plt
plt.quiver([0, 0], [0, 0], [1, -1], [1, 1], angles='xy', scale_units='xy', scale=1)
plt.xlim(-1.5, 1.5)
plt.xlabel("Input dimension 1")
plt.ylabel("Input dimension 2")
plt.ylim(-1.5, 1.5)
plt.show()