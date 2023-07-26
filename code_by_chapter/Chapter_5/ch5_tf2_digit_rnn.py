#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tom verguts
try out recurrent model (RNN or LSTM or ...) on the MNIST digits data
every row is treated as if it were a time point (so 28 time points)
and also 28 input features (n columns of a MNIST digit)
return_sequences = False meaning that only at the very last time step, feedback is expected
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def build_network(input_dim: int, output_dim: int, n_hid1: int, n_hid2: int, learning_rate: float = 0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape = (image_size, image_size)))
#    model.add(tf.keras.layers.SimpleRNN(n_hid1, return_sequences = False, activation = "tanh"))
    model.add(tf.keras.layers.LSTM(n_hid1, return_sequences = False, activation = "tanh"))
    model.add(tf.keras.layers.Dense(output_dim, activation = "softmax"))
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss, metrics = ["accuracy"])	
    return model	

def preprocess_digits(x_train, y_train,
                            train_size, x_test, y_test, test_size):
    n_labels = int(np.max(y_train)+1)
    image_size = x_train.shape[2]
    x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]
    x_train = x_train / 255   # 0-255 to 0-1
    x_test  = x_test  / 255   # same here
    y_train = tf.one_hot(y_train, n_labels)
    y_test  = tf.one_hot(y_test, n_labels)
    return image_size, n_labels, x_train, y_train, x_test, y_test

def train_model():
    res = model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))
    return res

def test_model(X, y):
    y_pred = model.predict(X)
    y_pred_label= np.argmax(y_pred, axis = 1) + 1
    return np.mean(y_pred_label == y)

def show_res(res, verbose: bool = False):
    if verbose:
        print(res.history)
    fig, axs = plt.subplots(2,2)
    axs[0, 0].set_title("training loss")
    axs[0, 0].plot(res.history["loss"])
    axs[0, 1].set_title("training accuracy")
    axs[0, 1].plot(res.history["accuracy"])
    axs[1, 0].set_title("test loss")
    axs[1, 0].plot(res.history["val_loss"])
    axs[1, 1].set_title("test accuracy")
    axs[1, 1].plot(res.history["val_accuracy"])
    plt.show()

# main program
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()   
    # pre-processing
    train_size, test_size = 10000, 50 # downscale to make data set smaller (and training faster)
    image_size, n_labels, X_train, y_train, X_test, y_test = preprocess_digits(
                                  x_train, y_train, train_size, x_test, y_test, test_size)

    model = build_network(input_dim = X_train.shape[1], output_dim = n_labels, n_hid1 = 20, n_hid2 = 10)
    res = train_model()
    show_res()
