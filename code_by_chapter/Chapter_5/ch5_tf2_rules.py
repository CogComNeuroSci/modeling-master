#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
does 3-layer network weight optimization via MSE minimization (i.e., backpropagation)
in TF2
for the logical rules AND, OR, or any other 2-valued logical rule (see MCP handbook for details of the example)
by default, the activation function is linear (argument activation = ...) in tf.keras.Sequential

"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initialize
np.set_printoptions(precision = 1, suppress = True)
learning_rate = 0.5
epochs    = 100 # how often to go through the whole data set
train_x   = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # a cat and a dog input pattern
test_x    = np.copy(train_x)                  # patterns to test the model after training
train_y   = np.array([0, 1, 1, 0])                 # a single output unit suffices for 2 categories
train_y   = train_y.reshape(4, 1)            # from a (2,) vector to a (2,1) matrix (not strictly needed)
test_y    = np.copy(train_y)
n_input, n_hidden, n_output  = train_x.shape[1], 6, 1

#%% construct the model
model = tf.keras.Sequential(layers = [
			tf.keras.Input(shape=(n_input,)),
 			tf.keras.layers.Dense(n_hidden, activation = "sigmoid"),
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
fig, ax = plt.subplots(1)
ax.plot(history.history["loss"], color = "black")

# train and test data mean error
to_test_x, to_test_y = [train_x, test_x], [train_y, test_y]
labels =  ["train", "test"]
print("\n")
for loop in range(2):
    y_pred = model.predict(to_test_x[loop])
    testdata_loss = tf.keras.losses.mean_squared_error(to_test_y[loop], y_pred)
    testdata_loss_summary = np.mean(testdata_loss.numpy())
    print("mean {} data error: {:.2f}".format(labels[loop], testdata_loss_summary))	

# actual outputs on the test data
print("\n")
print("predictions on the test data:")
print(test_data)


