#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom verguts
create fig 2.4
"""
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(start = -5, stop = 3, num = 20)
y = (x**4)/4 +(2/3)*(x**3)-(5/2)*(x**2)-6*x
plt.plot(x,y)
plt.ylabel("y = f(x)")