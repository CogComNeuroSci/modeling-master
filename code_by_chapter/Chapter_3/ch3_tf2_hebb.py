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
train_x = np.array([[1, 1, 0],
				    [0, 1, 1]])
train_t = np.array([[1, 0],
					[0, 1]]) # t for target
epochs = 10
learning_rate = 0.05

# define TensorFlow components
X = tf.Variable(initial_value = np.random.randn(1, train_x.shape[1]).astype(np.float32), name = "input")
T = tf.Variable(initial_value = np.random.randn(1, train_t.shape[1]).astype(np.float32), name = "output")
W = tf.Variable(initial_value = (np.random.randn(train_x.shape[1], train_t.shape[1])/100).astype(np.float32), name = "weights")

def cost():
    """ this cost function (eq (3.1) in MCP book) will be optimized"""
    return tf.matmul(-tf.matmul(X, W), tf.transpose(T)) 

opt  = tf.keras.optimizers.SGD(learning_rate = learning_rate) # SGD = stochastic gradient descent

for epoch in range(epochs):
    for (x, t) in zip(train_x, train_t):
        X.assign(x[np.newaxis,:])
        T.assign(t[np.newaxis,:])
        opt.minimize(cost, [W])
        if not epoch%1: # plot output only every 10 epochs
            w = W.numpy()
            print(w, '\n')
			
