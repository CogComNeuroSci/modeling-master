#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
written for TF2

image classification on the chicago face database
with a convolutional network (convnet)
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ch5_tf2_digit_classif import test_performance
from process_faces import load_faces_labels, show_face


# downscale to make data set smaller (and training faster)
train_size = 70
granularity = 5 # to downsample the image

x_train, y_train, x_test, y_test = load_faces_labels("CFD-Version-3.0/Images/CFD-INDIA", 
					 gran = granularity, n_faces = train_size, test = 0.3, depth = 3)
show_face(x_train[0:9], 3, 3, labels = y_train[0:9], title = "real labels") # plot random faces
learning_rate = 0.00002
epochs = 100
batch_size = 10
batches = int(x_train.shape[0] / batch_size)
stdev = 0.001

	  
#%% pre-processing
n_labels= int(np.max(y_train)+1)
x_train = x_train / 255   # to 0-1 data
y_train = tf.one_hot(y_train, n_labels) # one-hot coding (e.g., 3 becomes (0, 0, 0, 1, 0, ..))

#%% model definition
model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])),
			tf.keras.layers.AveragePooling2D(pool_size = (2, 2)),
			tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3)),
			tf.keras.layers.AveragePooling2D(pool_size = (2, 2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10, activation = "relu"),
			tf.keras.layers.Dense(10, activation = "relu"),
			tf.keras.layers.Dense(n_labels, activation = "softmax")])
model.build()

loss = tf.keras.losses.CategoricalCrossentropy()
#opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
opt = tf.keras.optimizers.legacy.Adam(learning_rate = learning_rate) # if you have an M1/M2 processor (like me)
model.compile(optimizer = opt, loss = loss)
#print("number of parameters= ", model.count_params())
#print(model.summary())
#%% model fitting
# plot before training
y_pred = model.predict(x_test)
show_face(x_test[0:9], 3, 3, labels = y_pred[0:9], title = "predictions before training") # plot faces before training
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

#%% show results
# error curve
y_pred = model.predict(x_test)
show_face(x_test[0:9], 3, 3, labels = y_pred[0:9], title = "predictions after training") # plot faces after training
fig = plt.figure()
plt.plot(history.history["loss"], color = "black")

test_performance(model, x_train, x_test, y_train, y_test)
