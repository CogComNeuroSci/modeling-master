#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tom verguts, edited by mehdi senoussi
Does cats-dogs network updating via minimization of activation function (2.3)

Solution to exercise 5
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initialize variables
x = np.array([1, 1, 0]) # prototypical cat
correct_winning_unit = 0
input_type = 'cat'

# x = np.array([0, 1, 1]) # prototypical dog
# correct_winning_unit = 1
# input_type = 'dog'

# W = np.array([[2., 1, 0], [0, 1, 2]])
W = np.array([[.5, .8, .2], [0.05, .6, 2]])

# net input to the cat and dog output units
in_all = np.matmul(W, x).reshape(2, 1).astype(np.float32)

# lateral inhibition between cat and dog
w_inh = -0.1
W_inh = w_inh * np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
update_rate = 0.1

# how many steps are we taking in the optimization process
### we increase the number of steps to make sure the model always finishes
epochs = 200
# create array to store values of output units (cat and dog) for each
# optimization step
y = np.zeros((epochs, 2))

## define TensorFlow components
# define a tf variable "Y" that represents the two output unit
Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32), name='Y')
# add noise at every time step to the Y values
add_noise = tf.compat.v1.assign(Y, Y + tf.random_normal((1, 2), mean = 0, stddev = 0.5))
# building the cost function, i.e. the energy function of our model that we 
# want to optimize
cost = tf.reduce_sum(-tf.matmul(Y, in_all) - tf.matmul(tf.matmul(Y,W_inh), tf.compat.v1.transpose(Y)))
# create an optimizer (that we will use to optimize the energy function)
opt  = tf.compat.v1.train.GradientDescentOptimizer(update_rate).minimize(cost)
# create an "initializer" that we will run to put actual values (numbers) in
# the TF variables
init = tf.global_variables_initializer()

# threshold to stop the optimization
threshold = 8

# define the number of trials and arrays to store the simulation results
n_trials = 20
model_accuracy = np.zeros(shape = (n_trials))
model_rt = np.zeros(shape = (n_trials))

for trial_n in np.arange(n_trials):
    print('trial {}'.format(trial_n))
    # run the graph
    with tf.compat.v1.Session() as sess:
        # initialize the network values
        sess.run(init)
        epoch = 0
        # loop over time steps until threshold is reached
        while np.all(y[epoch, :] < threshold):
            epoch += 1
            # run the noise addition to output variables (i.e. cat and dog units)
            sess.run(add_noise)
            # run the optimizer to 
            sess.run(opt, feed_dict = {})
            # calculate intermediate results
            y[epoch] = sess.run(Y, feed_dict = {})
    
    # get the winning output unit
    winning_unit = np.argmax(y[epoch]).squeeze()
    model_accuracy[trial_n] = int(correct_winning_unit == winning_unit)
    model_rt[trial_n] = epoch
    
    # # plot the cat / dog competition
    # plt.figure()
    # plt.plot(range(epochs), y[:,0], color = 'k', label='cat') # the cat
    # plt.plot(range(epochs), y[:,1], color = 'r', label='dog') # the dog
    # plt.vlines(x = epoch, ymin = y.min(), ymax = y.max()*1.1, linestyle='--',
    #            color=['k', 'r'][winning_unit])
    # plt.legend()

print('Model accuracy = {}%\nmodel RT median = {}, std = {}'.format(model_accuracy.mean()*100,
        np.median(model_rt), np.std(model_rt)))