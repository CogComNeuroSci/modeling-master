#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts
Does cats-dogs network updating via minimization of activation function (2.3)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initialize variables
#x_cat = np.array([1, 1, 0]) # prototypical cat
x_dog = np.array([0, 1, 1]) # prototypical dog
x = x_dog
W = np.array([[2, 1, 0], [0, 1, 2]]).astype(np.float32)
net = np.matmul(W, x).reshape(2,1).astype(np.float32) # net input to the cat and dog output units
w_inh = -0.1    # lateral inhibition between cat and dog
W_inh = w_inh*np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
learning_rate = 0.05
epochs = 100
y = np.ndarray((epochs, 2))

# define TensorFlow components
Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32), name="Y")

add_noise = tf.compat.v1.assign(Y, Y + tf.random_normal((1, 2), mean = 0, stddev = 0.5 ))
cost = tf.reduce_sum( -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.compat.v1.transpose(Y)) ) # the function to be optimized
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

# run the graph
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(add_noise) # to make the competition between cat and dog a bit more "noisy"
        sess.run(opt, feed_dict = {})
        if not epoch%1: # calculate intermediate results
            c = sess.run(cost, feed_dict = {})
            y[epoch] = sess.run(Y, feed_dict = {})
            #print("cost = {:.2f}, Y1 = {:.2f}, Y2 = {}".format(c, y[epoch][0], y[epoch][1]))

# plot the cat / dog competition
plt.plot(range(epochs), y[:,0], color = "k") # the cat
plt.plot(range(epochs), y[:,1], color = "r") # the dog        