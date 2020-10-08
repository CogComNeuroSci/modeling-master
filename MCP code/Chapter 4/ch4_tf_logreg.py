#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:46:46 2020

@author: This code is from Mark Jay
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(y_train.shape)
print(y_train[0])


#fig, axes = plt.subplots(1, 4, figsize=(7,3))
#for img, label, ax in zip(x_train[:4], y_train[:4], axes):
#    ax.set_title(label)
#    ax.imshow(img)
#    ax.axis("off")
#plt.show()

image_size = x_train.shape[1]*x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], image_size)  / 255
x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_test  = sess.run(tf.one_hot(y_test, 10))
    
learning_rate = 0.01
epochs = 10
batch_size = 100
batches = int(x_train.shape[0] / batch_size)

X = tf.placeholder(tf.float32, [None, image_size])
Y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(np.random.randn(image_size, 10).astype(np.float32))
B = tf.Variable(np.random.randn(10).astype(np.float32))

pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(pred), axis = 1))

opt  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for i in range(batches):
            offset = i * batch_size
            x = x_train[offset:(offset+batch_size)]
            y = y_train[offset:(offset+batch_size)]
            for (x_s, y_s) in zip(x, y):
                x_s = x_s.reshape(1,image_size)
                y_s = y_s.reshape(1,10)
                sess.run(opt, feed_dict= {X: x_s, Y: y_s})
        if not epoch % 20:
            c = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            print("cost {:.2f}".format(c))
        
        correct_pred = tf.equal(tf.math.argmax(pred, 1), tf.math.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc = accuracy.eval({X: x_test, Y: y_test})
        print(acc)
        
    fig, axes = plt.subplots(1, 6, figsize=(7,3))
    for (img, ax) in zip(x_test[:6], axes):
        img = img.reshape(1,image_size)
        y = np.argmax(sess.run(pred, feed_dict = {X: img}))
        ax.set_title(y)
        ax.imshow(img.reshape(28, 28))
        ax.axis("off")