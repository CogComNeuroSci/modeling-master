#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
(conv model code adapted from Sudharsan Ravichandiran; cats/dogs code from Pieter Huycke)
image classification; could a convolutional network solve this task...?
"""

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

# load cifar objects data
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# load MNIST numbers data
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# load cats and dogs data
objects = []
with (open("cats_dogs.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
dataset = objects[0]
x_train, y_train = dataset.images, dataset.target


# for piloting
# use x_train for x_test to check within-data fitting
x_train, y_train, x_test, y_test = x_train[:30,:], y_train[:30], x_train[:30,:], y_train[:30]

# pre-processing
if len(x_train.shape)>3:
    x_train, x_test = x_train[:,:,:,0], x_test[:,:,:,0]  # use first color channel only
y_train = y_train.astype(int)
y_test  = y_test.astype(int)
    

# for plotting
fig, axes = plt.subplots(1, 4, figsize=(7,3))
for img, label, ax in zip(x_train[:4], y_train[:4], axes):
    ax.set_title(label)
    ax.imshow(img, cmap = "gray")
    ax.axis("off")
plt.show()

# initialize model
image_size = x_train.shape[1:]
learning_rate = 0.01
epochs = 100
batch_size = 10
batches = int(x_train.shape[0] / batch_size)
n_labels = 2
filter_size = 5
k = 2       # size reduction in max pool layer

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = "SAME")

def reshape_x_batch(M):
    return M.reshape( (M.shape[0], M.shape[1]*M.shape[2]) )

def reshape_y_batch(M): # one-hot encoding
    if len(M.shape)>1:
        M = M[:, 0]         # remove a dimension
    b = np.zeros((M.size, n_labels))
    b[np.arange(M.size), M] = 1
    return b
    
w_c1 = tf.Variable(tf.random_normal([filter_size, filter_size, 1, 32]))
w_c2 = tf.Variable(tf.random_normal([filter_size, filter_size, 32, 64]))
b_c1 = tf.Variable(tf.random_normal([32]))
b_c2 = tf.Variable(tf.random_normal([64]))

x = tf.placeholder(tf.float32, [None, image_size[0]*image_size[1]])
x_reshaped = tf.reshape(x, [-1, image_size[0], image_size[1], 1])
y = tf.placeholder(tf.float32, [None, n_labels]) 

conv1 = tf.nn.relu(conv2d(x_reshaped, w_c1) + b_c1)
conv1 = maxpool2d(conv1)

conv2 = tf.nn.relu(conv2d(conv1, w_c2) + b_c2)
conv2 = maxpool2d(conv2)

x_flattened = tf.reshape(conv2, [-1, int(image_size[0]*image_size[1]/(k**4))*64])
w_fc = tf.Variable(tf.random_normal([int(image_size[0]*image_size[1]/(k**4)*64), 1024]))
b_fc = tf.Variable(tf.random_normal([1024]))
fc   = tf.nn.relu(tf.matmul(x_flattened, w_fc) + b_fc)

w_out = tf.Variable(tf.random_normal([1024, n_labels]))
b_out = tf.Variable(tf.random_normal([n_labels]))

output = tf.matmul(fc, w_out) + b_out
yhat = tf.nn.softmax(output)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))
opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run model
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for i in range(batches):
            offset = i * batch_size
            x_batch = reshape_x_batch(x_train[offset:(offset+batch_size)])
            y_batch = reshape_y_batch(y_train[offset:(offset+batch_size)])
            sess.run(opt, feed_dict= {x: x_batch, y: y_batch})
        if not epoch % 1:
            x_test_reshape = reshape_x_batch(x_test)
            y_test_reshape = reshape_y_batch(y_test)
            c = sess.run(cross_entropy, feed_dict={x: x_test_reshape, y: y_test_reshape})
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc = accuracy.eval({x: x_test_reshape, y: y_test_reshape})
            print("cost= {:.2f}, accuracy= {:.2f}".format(c, acc))        
