#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
written for TF2

image classification; could a standard three-layer network solve this task...?
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ch5_tf2_digit_classif import test_performance

def plot_pics(x_train, y_train):
    # plot some pictures from the data base
    fig, axes = plt.subplots(1, 4, figsize=(7,3))
    for img, label, ax in zip(x_train[:4], y_train[:4], axes):
        ax.set_title(label)
        ax.imshow(img)
        ax.axis("off")
    plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

plot_pics(x_train, y_train)

# for piloting, make a smaller data set
n_train_stim, n_test_stim = 1000, 100
x_train, y_train, x_test, y_test = x_train[:n_train_stim,:], y_train[:n_train_stim], x_test[:n_test_stim,:], y_test[:n_test_stim]

learning_rate = 0.0001
epochs = 2000
batch_size = 100
batches = int(x_train.shape[0] / batch_size)
stdev = 0.001
n_hid = 20

#%% pre-processing
n_labels = int(np.max(y_train)+1)
image_size = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
x_train = x_train.reshape(x_train.shape[0], image_size)  / 255
x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255
y_train = y_train[:,0] # remove a dimension
y_test  = y_test[:,0]
y_train = tf.one_hot(y_train, n_labels)
y_test  = tf.one_hot(y_test, n_labels)

	
# #%% model definition
model = tf.keras.Sequential([
 			tf.keras.Input(shape=(image_size,)),
 			tf.keras.layers.Dense(n_hid, activation = "relu"),
 			tf.keras.layers.Dense(n_labels, activation = "softmax")])
model.build()

loss = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = opt, loss = loss)

#%% run the model and show a summary of the results
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
model.summary()

#%% show results
# error curve
fig, ax = plt.subplots()
ax.plot(history.history["loss"], color = "black")

# print test data results
test_performance(model, x_train, x_test, y_train, y_test)
