#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom verguts
create figs of chapters 1
"""
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(start = -1, stop = 3, num = 20)
y = (x - 1)**2
plt.plot(x,y)
plt.ylabel("y = (x-1)^2")