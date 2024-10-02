#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mehdi senoussi

Makes a simple model composed of 2 input units and one output unit
to illustrate matrix multiplication (matmul) in TF2
"""

import tensorflow as tf
import numpy as np

# create TensorFlow constants of inputs (x1 and x2)
x = tf.constant(np.array([3.0, 0.3]).reshape(2, 1), name='x')
# create TensorFlow constants of weights
W = tf.constant(np.array([1.0, 3.0]).reshape(1, 2), name='W')

# create TensorFlow variable of output (y) as an operation on W and x
y = tf.matmul(W, x)

# print the tensor (works because TF2 uses eager execution)
print(y)

# transform to numpy
print(y.numpy())