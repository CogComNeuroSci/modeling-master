#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:25:08 2020

@author: tom verguts
simple function minimization with TensorFlow
"""

import tensorflow as tf
import numpy as np

epochs = 100
learning_rate = 0.1

X = tf.Variable(np.random.randn(1, 1).astype(np.float32), name="X")

cost = tf.reduce_sum((X-3)**2)
opt  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    x_previous= np.random.randn(1, 1)
    for epoch in range(epochs):
        sess.run(opt, feed_dict = {})
        x_previous = sess.run(X)
        print("x = {:.2f}".format(x_previous[0][0]))