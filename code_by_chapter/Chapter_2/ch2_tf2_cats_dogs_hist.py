#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:51:11 2020

@author: tom verguts
Does cats-dogs network updating via minimization of cost function (2.3)
the output is a histogram of the model response times
more features makes the prototypicality effect (in accuracy and RT) more pronounced
"""

import tensorflow as tf
from tensorflow.python.training import gradient_descent
import numpy as np
import matplotlib.pyplot as plt

# initialize variables
#x_cat = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0]) # prototypical cat
overlap = 0.9
x_dog = np.array([overlap, 1, 1, overlap, 1, 1, overlap, 1, 1]) # prototypical dog
x = x_dog/np.linalg.norm(x_dog) # in case you want to normalize the input vector
W = np.array([[2, 1, 0, 2, 1, 0, 2, 1, 0], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
net = np.matmul(W, x).reshape(2,1) # net input to the cat and dog output units
w_inh = -1    # lateral inhibition between cat and dog
W_inh = w_inh*np.array([[0, 1], [1, 0]])
step_size = 0.01
max_n_steps = 200
threshold = 5
ntrials = 1000
noise = 0.5
xmin, xmax = 0, max_n_steps # for easy comparison, always use the same x-range
RT = np.ndarray(ntrials)
accuracy = np.ndarray(ntrials)

# define TensorFlow components
Y  = tf.Variable(initial_value = np.random.randn(1, 2))


def cost():
	return -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.transpose(Y)) 


for trial_loop in range(ntrials):
	y = np.zeros((max_n_steps, 2))
	step = 0
	while (step < max_n_steps) and (np.max(y[step-(step>0)]) < threshold):
		Y.assign(Y + np.random.rand(1,2)*noise)
		y[step] = Y.numpy()
		gradient_descent.GradientDescentOptimizer(step_size).minimize(cost)
		Y.assign(tf.math.maximum(0, Y))
		step += 1
	RT[trial_loop] = step
	accuracy[trial_loop] = int(y[step-1, 0] < y[step-1, 1])
	Y.assign([[0, 0]])

# plot the  RT distribution
plt.hist(RT, density = True, color = "black", range = [xmin, xmax])

print("mean RT is {:.2f}".format(np.mean(RT)))
print("mean accuracy is {:.2%}".format(np.mean(accuracy)))