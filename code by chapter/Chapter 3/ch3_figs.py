#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:49:33 2018

@author: tom verguts
create plots from chapter 3 (supervised Hebb)
two orthonormal (i.e., orthogonal and length = 1) vectors
"""

import matplotlib.pyplot as plt
import numpy as np

v1, v2 = np.array([1, -1]), np.array([1, 1])
v1 = v1/np.linalg.norm(v1)
v2 = v2/np.linalg.norm(v2)

plt.quiver([0, 0], [0, 0], v1, v2, angles='xy', scale_units='xy', scale=1)
plt.xlim(-1.5, 1.5)
plt.xlabel("Input dimension 1")
plt.ylabel("Input dimension 2")
plt.ylim(-1.5, 1.5)
plt.show()