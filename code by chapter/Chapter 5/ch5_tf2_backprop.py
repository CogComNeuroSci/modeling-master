#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:37:36 2020

@author: tom verguts
Applies backprop via MSE minimization
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

model = tf.keras.Sequential([
			tf.keras.Input(shape=(2,)),
			tf.keras.layers.Dense(n_hid, activation = "relu"),
			tf.keras.layers.Dense(1)
			] )
model.build()
model.compile(optimizer = opt, loss=tf.keras.losses.MeanSquaredError())
history = model.fit(train_x, train_y, batch_size = 1, epochs = 10000, verbose = 0)
model.summary()

# error curve
plt.plot(history.history["loss"], color = "black")