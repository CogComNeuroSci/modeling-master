#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:19:23 2020

@author: tom verguts
Hebbian learning by cost minimization
"""
import tensorflow as tf
import numpy as np

# initialize  variables
train_x = np.array([[1, 0, 0], [0, 0, 1]])
train_y = np.array([[1, 0], [0, 1]])
epochs = 100
learning_rate = 0.1

# define TensorFlow components
X = tf.Variable(np.random.randn(1, 3).astype(np.float32), name = "weights")
Y = tf.Variable(np.random.randn(1, 2).astype(np.float32), name = "weights")
W = tf.Variable((np.random.randn(3, 2)/100).astype(np.float32), name = "weights")

def cost():
    return tf.matmul(-tf.matmul(X, W), tf.transpose(Y)) # this cost function will be optimized

opt  = tf.keras.optimizers.SGD(learning_rate = learning_rate)

for epoch in range(epochs):
    for (x, y) in zip(train_x, train_y):
        X.assign(x[np.newaxis,:])
        Y.assign(y[np.newaxis,:])
        opt.minimize(cost, [W])
        if not epoch%10:
            w = W.numpy()
            print(w)