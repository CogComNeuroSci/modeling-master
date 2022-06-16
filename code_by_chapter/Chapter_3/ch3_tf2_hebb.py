#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:19:23 2020

@author: tom verguts
Hebbian learning by cost minimization
look at the train_x and train_y patterns; can you predict what W will look like after training?
"""
import tensorflow as tf
import numpy as np

np.set_printoptions(precision = 2)
# initialize  variables
train_x = np.array([[1, 0, 0], [0, 0, 1]])
train_y = np.array([[1, 0], [0, 1]])
epochs = 20
learning_rate = 0.1

# define TensorFlow components
X = tf.Variable(np.random.randn(1, train_x.shape[1]).astype(np.float32), name = "input")
Y = tf.Variable(np.random.randn(1, train_y.shape[1]).astype(np.float32), name = "output")
W = tf.Variable((np.random.randn(train_x.shape[1], train_y.shape[1])/100).astype(np.float32), name = "weights")

def cost():
    return tf.matmul(-tf.matmul(X, W), tf.transpose(Y)) # this cost function (eq (3.1) in MCP book) will be optimized

opt  = tf.keras.optimizers.SGD(learning_rate = learning_rate)

for epoch in range(epochs):
    for (x, y) in zip(train_x, train_y):
        X.assign(x[np.newaxis,:])
        Y.assign(y[np.newaxis,:])
        opt.minimize(cost, [W])
        if not epoch%10: # plot output only every 10 epochs
            w = W.numpy()
            print(w, '\n')