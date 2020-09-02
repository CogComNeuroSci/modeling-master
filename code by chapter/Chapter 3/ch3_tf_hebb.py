#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:19:23 2020

@author: tom verguts
"""
import tensorflow as tf
import numpy as np

train_x = np.array([[1, 0, 0], [0, 0, 1]])
train_y = np.array([[1, 0], [0, 1]])
epochs = 100
learning_rate = 0.1

X = tf.compat.v1.placeholder(tf.float32, [None, 3])
Y = tf.compat.v1.placeholder(tf.float32, [None, 2])
W = tf.Variable(np.random.randn(3, 2).astype(np.float32), name = "weights")

Y_pred = tf.matmul(X, W)
cost = tf.reduce_sum(-tf.matmul(Y_pred, tf.transpose(Y)))
opt  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for (x, y) in zip(train_x, train_y):
            x = x.reshape(1, 3)
            y = y.reshape(1, 2)
            sess.run(opt, feed_dict = {X: x, Y: y})
            if not epoch%10:
                w = sess.run(W)
                print(w)