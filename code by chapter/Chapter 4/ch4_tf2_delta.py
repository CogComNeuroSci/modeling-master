#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
does 2-layer network weight optimization via MSE minimization
in TF2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1
epochs = 100
train_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
train_y = np.array([0, 1, 1, 1])
train_y = train_y.reshape(4,1)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(1))
model.build()
model.compile(optimizer = "adam", loss=tf.keras.losses.MeanSquaredError())
history = model.fit(train_x, train_y, batch_size = 1, epochs = 500)
model.summary()

# error curve
plt.plot(history.history["loss"], color = "black")