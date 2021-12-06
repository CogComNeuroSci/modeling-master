#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
written for TF2

digit classification with a convnet
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import digits dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# downscale to make data set smaller (and training faster)
train_size, test_size = 1000, 50
x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]

# plot some images from the data set
fig, axes = plt.subplots(1, 4, figsize=(7,3))
for img, label, ax in zip(x_train[:4], y_train[:4], axes):
      ax.set_title(label)
      ax.imshow(img)
      ax.axis("off")
	  
# pre-processing
n_labels = int(np.max(y_train)+1)
image_size = (x_train.shape[1], x_train.shape[2])
# x_train = x_train.reshape(x_train.shape[0], image_size)  / 255   # from 3D to 2D input data
# x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255   # same here
x_train = x_train / 255   # to 0-1 data
x_test  = x_test  / 255   # same here
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)   # from 3D to 2D input data
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)  # same here
y_train = tf.one_hot(y_train, n_labels)
y_test  = tf.one_hot(y_test, n_labels)


learning_rate = 0.0001
epochs = 1000
batch_size = 100
batches = int(x_train.shape[0] / batch_size)
stdev = 0.001

model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1)),
			tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (28, 28, 1)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10, activation = "relu"),
			tf.keras.layers.Dense(10, activation = "relu"),
			tf.keras.layers.Dense(n_labels, activation = "softmax")])
model.build()

loss = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = opt, loss = loss)
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
model.summary()

# error curve
fig = plt.figure()
plt.plot(history.history["loss"], color = "black")

# print test data results
to_test_x, to_test_y = [x_train, x_test], [y_train, y_test]
labels =  ["train", "test"]
print("\n")
for loop in range(2):
    y_pred = model.predict(to_test_x[loop])
    testdata_loss = tf.keras.losses.categorical_crossentropy(to_test_y[loop], y_pred)
    testdata_loss_summary = np.mean(testdata_loss.numpy())
    print("mean {} data performance: {:.2f}".format(labels[loop], testdata_loss_summary))	
