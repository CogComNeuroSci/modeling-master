#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:38:59 2021

@author: tom verguts
1-dimensional function minimization with TensorFlow2
"""
import tensorflow as tf
from tensorflow.python.training import gradient_descent

x = tf.Variable(10.0, trainable=True)


def f_x():
    return (x - 5)*(x - 5)


for _ in range(100):
    print([x.numpy(), f_x().numpy()])
    opt = gradient_descent.GradientDescentOptimizer(0.1).minimize(f_x)