#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:37:36 2020

@author: tom verguts
Applies backprop via Mean Squared Error minimization
with TF2 and keras
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)
n_hid = 5
train_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
train_y = np.array([0, 1, 1, 0])
train_y = train_y.reshape(4,1)

test_x, test_y = train_x, train_y # test data is just training data here

model = tf.keras.Sequential([
			tf.keras.Input(shape=(2,)),
			tf.keras.layers.Dense(n_hid, activation = "relu"),
			tf.keras.layers.Dense(1)
			] )
model.build()
model.compile(optimizer = opt, loss=tf.keras.losses.MeanSquaredError())
history = model.fit(train_x, train_y, batch_size = 1, epochs = 2000, verbose = 0)
model.summary()

# error curve
plt.plot(history.history["loss"], color = "black")

# print test data results
to_test_x, to_test_y = [train_x, test_x], [train_y, test_y]
labels =  ["train", "test"]
print("\n")
for loop in range(2):
    y_pred = model.predict(to_test_x[loop])
    testdata_loss = tf.keras.losses.mean_squared_error(to_test_y[loop], y_pred)
    testdata_loss_summary = np.mean(testdata_loss.numpy())
    print("mean {} data performance: {:.2f}".format(labels[loop], testdata_loss_summary))	