#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
does 2-layer network weight optimization via MSE minimization
in TF2
by default, activation = linear

Note: you can also do it more concisely:
model.compile(optimizer = "adam", loss=tf.keras.losses.MeanSquaredError())
but then you cannot specify learning rate explicitly
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1
epochs = 100
train_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
test_x = train_x
train_y = np.array([0, 1, 1, 1])
train_y = train_y.reshape(4,1)

model = tf.keras.Sequential([
			tf.keras.Input(shape=(2,)),
			tf.keras.layers.Dense(1, activation = "sigmoid")
			] )
model.build()

opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = opt, loss=tf.keras.losses.MeanSquaredError())
history = model.fit(train_x, train_y, batch_size = 1, epochs = epochs)
model.summary()
test_data = model.predict(test_x)

# error curve
plt.plot(history.history["loss"], color = "black")

# results on the test data
print("predictions on the test data:")
print(test_data)