#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
does 2-layer network weight optimization via MSE minimization
in TF2
for the cats vs dogs vs bagels example (example from a paper by James McClelland on prototype extraction;
example discussed more fully in McLeod, Plunkett, & Rolls, 1998)
by default, activation function = linear

"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# initialize
learning_rate = 0.005
epochs = 50 # how often to go through the whole training data set

np.set_printoptions(precision = 2, suppress = True)
filename = "cdb.npy"
prototype = np.load(filename)

n_train = 10
stim_dim = prototype.shape[1]
tot_n_train = prototype.shape[0]*n_train
train_x = np.ndarray((tot_n_train, stim_dim))

test_x = prototype                           # patterns to test the model after training
train_y = np.array((np.zeros(n_train), np.zeros(n_train) + 1, np.zeros(n_train) + 2))
train_y = train_y.reshape(tot_n_train, 1)            # from a (tot_n_train,) vector to a (tot_n_train,1) matrix 
print(train_y.shape)
train_y = tf.one_hot(train_y, 3)

#%% helper functions
# generate a perturbation of the prototype
# variable "same" indicates whether it is allowed for the perturbation to equal the prototype
# prob indicates the probability of a perturbation in any single element
def perturbation(proto, prob = 0.05, same = True):
    proposal = np.copy(proto)
    done = False
    while not done:
        for loop in range(stim_dim):
            proposal[loop] = np.random.choice([-1, +1], p = [prob, 1-prob]) * proto[loop]
        if (same == True) or (np.any(proposal != proto)):
            done = True
    return proposal

for loop in range(prototype.shape[0]):
	for train_loop in range(n_train):
		train_x[loop*n_train + train_loop, :] = perturbation(prototype[loop], prob = 0.2, same = False)


#%% construct the model
model = tf.keras.Sequential(layers = [
 			tf.keras.Input(shape=(stim_dim,)),
 			tf.keras.layers.Dense(3, activation = "sigmoid") # Dense?... remember the convolutional network?
 			] )
model.build()

#%% train & test the model
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)           # Adam is a kind of gradient descent
model.compile(optimizer = opt, loss=tf.keras.losses.MeanSquaredError()) # loss is what we called energy
history = model.fit(train_x, train_y, batch_size = 1, epochs = epochs)
model.summary()
test_data = model.predict(test_x)


#%% report data
# train data: error curve
fig, axs = plt.subplots(1, 2)
axs[0].plot(history.history["loss"], color = "black")

# test data
print("predictions on the test data:")
print(test_data)
axs[1].imshow(test_data)
axs[1].set_title("confusion matrix") # prints the confusion between cats (row & column 0), dogs (row & column 1), and bagels (row & column 2)
