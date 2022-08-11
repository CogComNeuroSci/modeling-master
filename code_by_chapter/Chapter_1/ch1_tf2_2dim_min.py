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

n_steps = 10
update_rate = 0.01
offsets = [2, -2]

X = tf.Variable(initial_value = np.random.randn(1, 2))

def f_x():
    return (X[0, 0] - offsets[0])*(X[0, 0] - offsets[0]) + (X[0, 1] - offsets[1])*(X[0, 1] - offsets[1])

for _ in range(n_steps):
    print("X = ({:.2f}, {:.2f}), f(X) = {:.2f}".format(X.numpy()[0, 0], X.numpy()[0, 1], f_x().numpy()))
    gradient_descent.GradientDescentOptimizer(update_rate).minimize(f_x)