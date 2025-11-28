#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
written for TF2

MNIST digit classification
with a convolutional network (convnet)
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ch5_tf2_digit_classif import plot_digits, test_performance

# import digits dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# downscale to make data set smaller (and training faster)
train_size, test_size = 1000, 50
x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]

learning_rate = 0.0001
epochs = 100
batch_size = 100
batches = int(x_train.shape[0] / batch_size)
stdev = 0.001

plot_digits(x_train[:4], y_train[:4])
	  
#%% pre-processing
n_labels = int(np.max(y_train)+1)
x_train = x_train / 255   # to 0-1 data
x_test  = x_test  / 255   # same here
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)   # from 3D to 4D input data for the convnet
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)       # same here
y_train = tf.one_hot(y_train, n_labels) # one-hot coding (e.g., 3 becomes (0, 0, 0, 1, 0, ..))
y_test  = tf.one_hot(y_test, n_labels)

#%% model definition
model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])),
			tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10, activation = "relu"),
			tf.keras.layers.Dense(10, activation = "relu"),
			tf.keras.layers.Dense(n_labels, activation = "softmax")])
model.build()

loss = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = opt, loss = loss)

#%% model fitting
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
model.summary()

#%% show results
# error curve
fig = plt.figure()
plt.plot(history.history["loss"], color = "black")

test_performance(model, x_train, x_test, y_train, y_test)
