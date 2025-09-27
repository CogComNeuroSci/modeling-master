#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:49 2020

@author: tom verguts
written for TF2

image classification on the CIFAR-10 data;
could a standard three-layer network solve this task...?
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ch5_tf2_digit_classif import test_performance

def plot_imgs(x_train, y_train):
    """plot some pictures from the data base"""
    labels = ["airplane", "automobile", "bird", "cat",
		   "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, axes = plt.subplots(1, 4, figsize=(7,3))
    for img, label, ax in zip(x_train[:4], y_train[:4], axes):
        ax.set_title(labels[int(label)])
        ax.imshow(img)
        ax.axis("off")
    plt.show()

def preprocess_imgs(x_train, y_train, train_size, x_test, y_test, test_size,
					   image_size, n_labels):
    x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]
    x_train = x_train.reshape(x_train.shape[0], image_size)  / 255
    x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255
    y_train = y_train[:,0] # remove a dimension
    y_test  = y_test[:,0]
    y_train = tf.one_hot(y_train, n_labels)
    y_test  = tf.one_hot(y_test, n_labels)
    return x_train, y_train, x_test, y_test

# %% main code
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_original = np.copy(x_train)
    plot_imgs(x_train, y_train)
    # for piloting, make a smaller data set
    n_train_stim, n_test_stim = 2000, 1000
    learning_rate = 0.00005
    epochs = 10000
    batch_size = 100
    batches = int(x_train.shape[0] / batch_size)
    stdev = 0.001
    n_hid = 10

    # pre-processing
    n_labels = int(np.max(y_train)+1)
    image_size = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
    x_train, y_train, x_test, y_test = preprocess_imgs(
		                                  x_train, y_train, n_train_stim, x_test, y_test, n_test_stim, image_size = image_size, n_labels = n_labels)
	
    # model definition
    model = tf.keras.Sequential([
 			tf.keras.Input(shape=(image_size,)),
 			tf.keras.layers.Dense(n_hid, activation = "relu"),
 			tf.keras.layers.Dense(n_labels, activation = "softmax")])
    model.build()

    loss = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer = opt, loss = loss)

    # run the model and show a summary of the results
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
    model.summary()

    # show results
    # error curve
    fig, ax = plt.subplots()
    y_pred = np.argmax(model(x_train).numpy(), axis = 1)
    plot_imgs(x_train_original, y_pred)
    ax.plot(history.history["loss"], color = "black")

    # print test data results
    test_performance(model, x_train, x_test, y_train, y_test) 