#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:07:08 2018
Do the classical tasks...
with sigmoid transfer function
@author: tom
"""

import tensorflow as tf
import numpy as np

# input data
n_train = 1000
X_in = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = np.array([0, 0, 0, 1])

# construct model
w = tf.Variable(dtype = tf.float32, initial_value = tf.random_normal((2,1)))
b = tf.Variable(dtype = tf.float32, initial_value = tf.random_normal((1,1)))
x = tf.placeholder(tf.float32, name = "x")
y = tf.placeholder(tf.float32, name = "y")
y_model = tf.sigmoid(tf.add(tf.matmul(x,w),b))
y_model= tf.reshape(y_model,[-1])
error = tf.pow(y-y_model,2, name = "error")
loss_function = tf.reduce_mean(error, name = "loss_function")
learn = tf.train.AdamOptimizer(0.5).minimize(loss_function)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("output_tf_learn2", sess.graph)
    cost_summary = tf.summary.scalar("cost", loss_function)
    for i in range(n_train):
        sess.run(learn, {x: X_in, y: Y})
        summary = sess.run(cost_summary, feed_dict = {x: X_in, y: Y})
        writer.add_summary(summary,i)
    y_pred = sess.run(y_model, {x: X_in})
    print("predictions: {}".format(y_pred))
    print("weights: {} {}".format(sess.run(w), sess.run(b)))
    print("error: {}".format(sess.run(loss_function, {x: X_in, y: Y})))