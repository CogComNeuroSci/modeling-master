#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
image classification; could a standard three-layer network solve this task...?
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# for plotting
#fig, axes = plt.subplots(1, 4, figsize=(7,3))
#for img, label, ax in zip(x_train[:4], y_train[:4], axes):
#    ax.set_title(label)
#    ax.imshow(img)
#    ax.axis("off")
#plt.show()

# pre-processing
n_labels = int(np.max(y_train)+1)
image_size = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
x_train = x_train.reshape(x_train.shape[0], image_size)  / 255
x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255
y_train = y_train[:,0] # remove a dimension
y_test  = y_test[:,0]

# for piloting
x_train, y_train, x_test, y_test = x_train[:10000,:], y_train[:10000], x_test[:100,:], y_test[:100]

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, n_labels))
    y_test  = sess.run(tf.one_hot(y_test, n_labels))

learning_rate = 0.001
epochs = 1000
batch_size = 100
batches = int(x_train.shape[0] / batch_size)
stdev = 0.001
n_hid = 10

X   = tf.placeholder(tf.float32, [None, image_size])
Y   = tf.placeholder(tf.float32, [None, n_labels])
W1  = tf.Variable(np.random.randn(image_size, n_hid).astype(np.float32)*stdev)
W2  = tf.Variable(np.random.randn(n_hid, n_labels).astype(np.float32)*stdev)
B   = tf.Variable(np.random.randn(n_labels).astype(np.float32))

hid  = tf.matmul(X, W1)
hidT = 1/(1+tf.math.exp(-hid)) # hidden transformed : Note: this is not softmax
pred = tf.nn.softmax(tf.add(tf.matmul(hidT, W2), B))
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(pred), axis = 1))
opt  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for i in range(batches):
            offset = i * batch_size
            x = x_train[offset:(offset+batch_size)]
            y = y_train[offset:(offset+batch_size)]
            for (x_s, y_s) in zip(x, y):
                x_s = x_s.reshape(1,image_size)
                y_s = y_s.reshape(1,n_labels)
                sess.run(opt, feed_dict= {X: x_s, Y: y_s})
        if not epoch % 5:
            for loop in range(2):
                if loop == 0:
                    data_x, data_y = x_train, y_train
                else:
                    data_x, data_y = x_test, y_test
                c = sess.run(cost, feed_dict={X: data_x, Y: data_y})
                correct_pred = tf.equal(tf.math.argmax(pred, 1), tf.math.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                acc = accuracy.eval({X: data_x, Y: data_y})
                print("{} cost= {:.2f}, accuracy= {:.2f}".format(["train", "test"][loop], c, acc))        
            print("\n")   
