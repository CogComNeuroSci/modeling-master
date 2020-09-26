#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts
Does activation updating via minimization of activation function (2.3)
"""

import tensorflow.compat.v1 as tf
import numpy as np

# initialize variables
x = np.array([0, 1, 1]) # input units
W = np.array([[2, 0, 0], [0, 0, 2]]).astype(np.float32) # transformation matrix x to net input
net = np.matmul(W, x).reshape(2,1).astype(np.float32)   # matrix multiplication (matmul)
w_inh = -0.1 # inhibition between output (Y) units
W_inh = w_inh*np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
learning_rate = 0.1
epochs = 10

# TensorFlow components
Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32), name="Y") # this will be optimized to minimize cost

# next line defines the energy function that will be minimized
# the reduce_sum is only to turn an array into a float; also tf.add could be used (as in the commented code), but then it remainns an array
cost       = tf.reduce_sum( -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.transpose(Y)) )
#cost       = tf.add( -tf.matmul(Y,net), - tf.matmul( tf.matmul(Y,W_inh), tf.transpose(Y)) )
optimizer  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init       = tf.global_variables_initializer()

# run the graph
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(optimizer, feed_dict = {})
        if not epoch%2: # how often to show intermediate results?
            c = sess.run(cost, feed_dict = {})
            y = sess.run(Y, feed_dict = {})
            print("cost = {:.2f}, Y1 = {:.2f}, Y2 = {}".format(c, y[0][0], y[0][1]))
        