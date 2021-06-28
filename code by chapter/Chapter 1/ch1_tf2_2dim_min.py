#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:25:08 2020

@author: tom verguts
2-dimensional function minimization with TensorFlow2
"""

import tensorflow as tf
from tensorflow.python.training import gradient_descent
import numpy as np

epochs = 10
update_rate = 0.1
offsets = [2, -2]

X = tf.Variable(np.random.randn(1, 2).astype(np.float32), name="X")

def f_x():
    return (X[0, 0] - offsets[0])*(X[0, 0] - offsets[0]) + (X[0, 1] - offsets[1])*(X[0, 1] - offsets[1])

for _ in range(100):
    print([X.numpy(), f_x().numpy()])
    opt = gradient_descent.GradientDescentOptimizer(update_rate).minimize(f_x)