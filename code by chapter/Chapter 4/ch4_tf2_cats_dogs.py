#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
does 2-layer network weight optimization via MSE minimization
in TF2
for the cats vs dogs example (see handbook for details of the example)
by default, activation = linear

Note: you can also do it more concisely:
model.compile(optimizer = "adam", loss=tf.keras.losses.MeanSquaredError())
but then you cannot specify learning rate explicitly
"""

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initialize
learning_rate = 0.05
epochs = 100 # how often to go through the whole data set
train_x = np.array([[1, 1, 0], [0, 1, 1]]) # a cat and a dog input pattern
test_x = train_x                           # patterns to test the model after training
train_y = np.array([0, 1])                 # a single output unit suffices for 2 categories
train_y = train_y.reshape(2, 1)            # from a (2,) vector to a (2,1) matrix (not strictly needed)

# construct the model
model = tf.keras.Sequential(layers = [
 			tf.keras.Input(shape=(3,)),
 			tf.keras.layers.Dense(1, activation = "relu") # Dense?... remember the convolutional network?
 			] )
model.build()

# train & test the model
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)           # Adam is a kind of gradient descent
model.compile(optimizer = opt, loss=tf.keras.losses.MeanSquaredError()) # loss is what we called energy
history = model.fit(train_x, train_y, batch_size = 1, epochs = epochs)
model.summary()
test_data = model.predict(test_x)


# report data
# train data: error curve
plt.plot(history.history["loss"], color = "black")

# test data
print("predictions on the test data:")
print(test_data)