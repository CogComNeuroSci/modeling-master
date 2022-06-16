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
#x_cat = np.array([1, 1, 0]) # prototypical cat
x_dog = np.array([0.8, 1, 1, 0.8, 1, 1]) # prototypical dog
x = x_dog #/np.linalg.norm(x_dog)
W = np.array([[2, 1, 0, 2, 1, 0], [0, 1, 2, 0, 1, 2]]).astype(np.float32)
net = np.matmul(W, x).reshape(2,1).astype(np.float32) # net input to the cat and dog output units
w_inh = -0.2    # lateral inhibition between cat and dog
W_inh = w_inh*np.array([[0, 1], [1, 0]])
W_inh = W_inh.astype(np.float32)
step_size = 0.05
max_n_steps = 200
threshold = 50
ntrials = 200
noise = 2
RT = np.ndarray(ntrials)
accuracy = np.ndarray(ntrials)

# define TensorFlow components
Y  = tf.Variable(np.random.randn(1, 2).astype(np.float32), name="Y")


def cost():
	return -tf.matmul(Y,net) - tf.matmul(tf.matmul(Y,W_inh), tf.transpose(Y)) 


for trial_loop in range(ntrials):
	y = np.zeros((max_n_steps, 2))
	step = 0
	while (step < max_n_steps) and (np.max(y[step-(step>0)]) < threshold):
		Y.assign(Y + np.random.rand(1,2)*noise)
		y[step] = Y.numpy()
		opt = gradient_descent.GradientDescentOptimizer(step_size).minimize(cost)
		Y.assign(tf.math.maximum(0, Y))
		step += 1
	RT[trial_loop] = step
	accuracy[trial_loop] = int(y[step-1, 0] < y[step-1, 1])
	Y.assign([[0, 0]])

# plot the  RT distribution
plt.hist(RT, color = "black")

print("mean RT is {:.2f}".format(np.mean(RT)))
print("mean accuracy is {:.2%}".format(np.mean(accuracy)))