#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
does 3-layer network weight optimization via MSE minimization
in TF2
for the logical rules AND, OR, or any other 2-valued logical rule (see MCP handbook for details of the example)
by default, the activation function is linear (argument activation = ...) in tf.keras.Sequential

TO BE UPDATED!
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initialize
learning_rate = 0.5
epochs    = 100 # how often to go through the whole data set
train_x   = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # a cat and a dog input pattern
test_x    = np.copy(train_x)                  # patterns to test the model after training
train_y   = np.array([0, 1, 1, 0])                 # a single output unit suffices for 2 categories
train_y   = train_y.reshape(4, 1)            # from a (2,) vector to a (2,1) matrix (not strictly needed)
n_hidden, n_output  = 2, 1

#%% construct the model
model = tf.keras.Sequential(layers = [
 			tf.keras.Input(shape=(n_hidden,)),
 			tf.keras.layers.Dense(n_output, activation = "sigmoid") # Dense?... remember the convolutional network?
 			] )
model.build()

#%% train & test the model
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)           # Adam is a kind of gradient descent
model.compile(optimizer = opt, loss=tf.keras.losses.MeanSquaredError()) # loss is what we called energy
history = model.fit(train_x, train_y, batch_size = 1, epochs = epochs)
model.summary()
test_data = model.predict(test_x)


#%% report data
print(model.get_weights())
# train data: error curve
plt.plot(history.history["loss"], color = "black")

# test data
print("predictions on the test data:")
print(test_data)


