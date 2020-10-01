#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts, edited by mehdi senoussi
Does cats-dogs network updating via minimization of activation function (2.3)
"""

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

# initialize variables
x = np.array([1, 1, 0]) # prototypical cat
input_type = 'cat'

# x = np.array([0, 1, 1]) # prototypical dog
# input_type = 'dog'


W = np.array([[1., .5, .1], [.1, .5, 1]])

# net input to the cat and dog output units
in_all = np.matmul(W, x).reshape(2, 1).astype(np.float32)

# lateral inhibition between cat and dog
w_inh = -0.1
W_inh = w_inh * np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
update_rate = 0.05

# how many steps are we taking in the optimization process
epochs = 100
# create array to store values of output units (cat and dog) for each
# optimization step
y = np.zeros((epochs, 2))

## define TensorFlow components
# define a tf variable "Y" that represents the two output unit
Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32), name='Y')
# add noise at every time step to the Y values
add_noise = tf.assign(Y, Y + tf.random_normal((1, 2), mean = 0, stddev = 0.5))
# building the cost function, i.e. the energy function of our model that we 
# want to optimize
cost = tf.reduce_sum(-tf.matmul(Y, in_all) - tf.matmul(tf.matmul(Y,W_inh), tf.transpose(Y)))
# create an optimizer (that we will use to optimize the energy function)
opt  = tf.train.GradientDescentOptimizer(update_rate).minimize(cost)
# create an "initializer" that we will run to put actual values (numbers) in
# the TF variables
init = tf.global_variables_initializer()

# run the graph
with tf.Session() as sess:
    # initialize the network values
    sess.run(init)
    # loop over all time steps
    for epoch in range(epochs):
        # run the noise addition to output variables (i.e. cat and dog units)
        sess.run(add_noise)
        # run the optimizer to 
        sess.run(opt, feed_dict = {})
        # calculate intermediate results
        y[epoch] = sess.run(Y, feed_dict = {})


# plot the cat / dog competition
plt.figure()
plt.title('input: {}'.format(input_type))
plt.plot(range(epochs), y[:,0], color = 'k', label='cat') # the cat
plt.plot(range(epochs), y[:,1], color = 'r', label='dog') # the dog
plt.legend()

