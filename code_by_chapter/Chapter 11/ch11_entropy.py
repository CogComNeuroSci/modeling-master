#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 09:53:37 2022

@author: tom verguts
entropy function as a function of p; should normally be with 2-log,
but this is just a constant different
"""
import numpy as np
import matplotlib.pyplot as plt

p = np.arange(0.01, 1, 0.01)
entropy = -np.multiply(p, np.log(p)) - np.multiply(1-p, np.log(1-p)) 
plt.plot(p, entropy, color = "black")
plt.xlabel("p")
plt.ylabel("Entropy")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)