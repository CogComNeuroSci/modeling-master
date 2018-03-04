#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:07:08 2018
Do the classical tasks...
with sigmoid transfer function
now with hidden units and some scope
@author: tom
"""

import tensorflow as tf
import numpy as np
tf.reset_default_graph()

# input data
n_train = 1000
X_in = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Z = np.array([0, 1, 1, 1])
n_hid = tf.constant(3)
sessionnr = input("session nr? ")
logs_path = "output_tf_learn3/session"+str(sessionnr)

# construct model
with tf.variable_scope("layer1"):
    w1 = tf.Variable(dtype = tf.float32, initial_value = tf.random_normal((2,n_hid)), name = "w1")
    b1 = tf.Variable(dtype = tf.float32, initial_value = tf.random_normal((1,1)), name = "bias1")
    x = tf.placeholder(tf.float32, name = "x")
    y = tf.sigmoid(tf.add(tf.matmul(x,w1),b1))

with tf.variable_scope("layer2"):
    w2 = tf.Variable(dtype = tf.float32, initial_value = tf.random_normal((n_hid,1)), name = "w2")
    b2 = tf.Variable(dtype = tf.float32, initial_value = tf.random_normal((1,1)), name = "bias2")
    z = tf.placeholder(tf.float32, name = "z") # targets dus
    z_model = tf.sigmoid(tf.add(tf.matmul(y,w2),b2))
    z_model= tf.reshape(z_model,[-1])
    with tf.variable_scope("gradient_descent"):
        error = tf.pow(z-z_model,2, name = "error")
        loss_function = tf.reduce_mean(error, name = "loss_function")
        learn = tf.train.AdamOptimizer(0.5).minimize(loss_function)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path, sess.graph)
    cost_summary = tf.summary.scalar("cost", loss_function)
    for i in range(n_train):
        sess.run(learn, {x: X_in, z: Z})
        summary = sess.run(cost_summary, feed_dict = {x: X_in, z: Z})
        writer.add_summary(summary,i)
    z_pred = sess.run(z_model, {x: X_in})
    print("predictions: {}".format(z_pred))
    print("weights: {} {}".format(sess.run(w1), sess.run(b1)))
    print("error: {}".format(sess.run(loss_function, {x: X_in, z: Z})))
    writer.close()