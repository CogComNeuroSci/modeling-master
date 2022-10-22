#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:19:23 2020
@author: tom verguts
Hebbian learning by cost minimization
look at the train_x and train_y patterns; can you predict what W will look like after training?
simplified version with just a single tensor
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.training import gradient_descent

np.set_printoptions(precision = 2)
# initialize  variables
train_x = np.array([[1.0, 1.0, 0.0], [0., 1., 1.]])
train_t = np.array([[1., 0.], [0., 1.]]) # t for target
epochs = 30
learning_rate = 0.1

# define numpy and TensorFlow components
X = np.random.randn(1, train_x.shape[1]) # 1, 3
T = np.random.randn(1, train_t.shape[1]) # 1, 2
W = tf.Variable( initial_value = (np.random.randn(train_x.shape[1], train_t.shape[1])/100) ) # 3, 2

def cost():
    return tf.matmul(-tf.matmul(X, W), tf.transpose(T)) # this cost function (eq (3.1) in MCP book) will be optimized

for epoch in range(epochs):
    for (x, t) in zip(train_x, train_t):
        X = x[np.newaxis,:]  # X has now size (1, 3); not just (,3)
        T = t[np.newaxis,:]
        gradient_descent.GradientDescentOptimizer(learning_rate).minimize(cost) # core of the code
        if not epoch%10: # plot output only every 10 epochs
            w = W.numpy()
            print(w, '\n')
