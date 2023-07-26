#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
written for TensorFlow 2

image classification on the CIFAR-10 data set;
could a two-layer network solve this task...?
note that this file uses some functions from ch5_tf2_digit_classif,
and from ch5_tf2_image_classif, so you must put those files in the 
working directory (or in directory X and and point sys.path.append
to that directory X, as done here)
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tom/Documents/Modcogproc/modeling-master/code_by_chapter/Chapter_5')
from ch5_tf2_digit_classif import test_performance
from ch5_tf2_image_classif import plot_imgs

# images dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# to make data set smaller (and training faster)
x_train, y_train, x_test, y_test = x_train[:500,:], y_train[:500], x_test[:500,:], y_test[:500]

# estimation parameters
learning_rate = 0.0001
epochs = 1000 # how often to go through the whole data set
batch_size = 100
batches = int(x_train.shape[0] / batch_size)

plot_imgs(x_train, y_train)

#%% pre-processing
n_labels = int(np.max(y_train)+1)
image_size = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
x_train = x_train.reshape(x_train.shape[0], image_size)  / 255
x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255
y_train = y_train[:,0] # remove a dimension
y_test  = y_test[:,0]  
y_train = tf.one_hot(y_train, n_labels)
y_test  = tf.one_hot(y_test, n_labels)

#%% model construction
model = tf.keras.Sequential([
			tf.keras.Input(shape=(image_size,)),
			tf.keras.layers.Dense(n_labels, activation = "softmax")
			] )
model.build()

loss = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = opt, loss = loss)

#%% fit the model; this part should take the longest
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
model.summary()

#%% report results
# error curve
fig, ax = plt.subplots(1, figsize=(7,3))
ax.plot(history.history["loss"], color = "black")

# print test data results
test_performance(model, x_train, x_test, y_train, y_test)