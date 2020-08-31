#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:37:36 2020

@author: tom verguts
does 2-layer network weight optimization via MSE minimization
"""

import tensorflow as tf
import numpy as np

learning_rate = 0.1
epochs = 10
train_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
train_y = np.array([0, 1, 1, 0])
train_y = train_y.reshape(4,1)

X = tf.compat.v1.placeholder(tf.float32, [None, 2])
Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
W = tf.Variable(np.random.randn(2,1).astype(np.float32), name="weights")
B = tf.Variable(np.random.randn(1).astype(np.float32), name="bias")

pred = 1/(1+tf.math.exp(tf.add(tf.matmul(X, W), B))) #1D softmax
cost = tf.reduce_sum((pred - Y)**2)
opt  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        ix = np.random.permutation(range(train_x.shape[0]))
        train_x = train_x[ix]
        train_y = train_y[ix]
        for (x, y) in zip(train_x, train_y):
            x = x.reshape(1, 2)
            y = y.reshape(1, 1)
            sess.run(opt, feed_dict = {X: x, Y: y})
        if not epoch%100:
            w = sess.run(W)
            b = sess.run(B)
            c = sess.run(cost, feed_dict = {X: train_x, Y: train_y})
            print("cost = {:.2f}".format(c))
            