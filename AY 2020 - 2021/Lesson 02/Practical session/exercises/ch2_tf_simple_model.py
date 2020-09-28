#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mehdi senoussi

Makes a simple model composed of 2 input units and one output unit
"""

import tensorflow as tf
import numpy as np

# create TensorFlow constants of inputs (x1 and x2)
x = tf.constant(np.array([1.0, 2.0]).reshape(2, 1), name='x')
# create TensorFlow constants of weights
W = tf.constant(np.array([1.0, 3.0]).reshape(1, 2), name='W')

# create TensorFlow variable of output (y) as an operation on W and x
y = tf.matmul(W, x)

# setup the variable initialisation
init = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init)
    # compute the output of the graph
    y_out = sess.run(y)
    print("Variable y is {}".format(y_out))