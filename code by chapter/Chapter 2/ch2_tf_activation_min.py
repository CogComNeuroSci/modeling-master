#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts
Does activation updating via minimization of activation function (2.3)
"""

import tensorflow as tf
import numpy as np

x = np.array([0, 1, 1])
W = np.array([[2, 0, 0], [0, 0, 2]]).astype(np.float32)
net = np.matmul(W, x).reshape(2,1).astype(np.float32)
w_inh = -0.1
W_inh = w_inh*np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
learning_rate = 0.1
epochs = 10

Y0 = tf.Variable(np.random.randn(1, 2).astype(np.float32), name="Y0")
Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32), name="Y")

cost = tf.reduce_sum( -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.compat.v1.transpose(Y)) )
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(opt, feed_dict = {})
        if not epoch%2:
            c = sess.run(cost, feed_dict = {})
            y = sess.run(Y, feed_dict = {})
            print("cost = {:.2f}, Y1 = {:.2f}, Y2 = {}".format(c, y[0][0], y[0][1]))
        