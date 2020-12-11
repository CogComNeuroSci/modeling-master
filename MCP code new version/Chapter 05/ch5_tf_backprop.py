#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:37:36 2020

@author: tom verguts
Applies backprop via MSE minimization
"""

import tensorflow as tf
import numpy as np

learning_rate = 0.1
nhid = 3
epochs = 10000
train_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
train_y = np.array([0, 1, 1, 0])
train_y = train_y.reshape(4,1)

X  = tf.compat.v1.placeholder(tf.float32, [None, 2])
Y  = tf.compat.v1.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(np.random.randn(2,nhid).astype(np.float32), name="weights1")
W2 = tf.Variable(np.random.randn(nhid,1).astype(np.float32), name="weights2")
B1 = tf.Variable(np.random.randn(nhid).astype(np.float32),   name="bias")
B2 = tf.Variable(np.random.randn(1).astype(np.float32),      name="bias")

hid =  1/(1+tf.math.exp(tf.add(tf.matmul(X, W1), B1)))
pred = 1/(1+tf.math.exp(tf.add(tf.matmul(hid, W2), B2)))
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
            w1 = sess.run(W1)
            c = sess.run(cost, feed_dict = {X: train_x, Y: train_y})
            print("cost = {}".format(c))
            