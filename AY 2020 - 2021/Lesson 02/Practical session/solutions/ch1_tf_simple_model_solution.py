#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mehdi senoussi
"""

import tensorflow as tf
import numpy as np

# create TensorFlow constants of inputs (x1 and x2)
### we reshape the array so that its shape allows to do dot product with
### the weights array
x = tf.constant(np.array([1.0, 2.0, 0.5]).reshape(3, 1), name='x')

# create TensorFlow constants of weights
### here we did not need to reshape the array because its size/dimension is
### in the correct format as there are more than 1 row and one column but we
### keep the reshape for consistency
W = tf.constant(np.array([[1.0, 3.0, 2.0], [2.0, 1.0, 0.5]]).reshape(2, 3), name='W')

# create TensorFlow variable of output (y) as an operation on W and x
### we changed the name of this variable to represent the fact that it will
### now hold the two y values
y_all = tf.matmul(W, x)

# setup the variable initialisation
init = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init)
    # compute the output of the graph
    y_out = sess.run(y_all)
    print("Variables y1 is {} and y2 is {}".format(y_out[0], y_out[1]))