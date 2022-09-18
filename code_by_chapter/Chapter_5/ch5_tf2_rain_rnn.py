#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tom verguts
try out recurrent model (RNN or LSTM or ...) on rain data from kaggle
"""

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/tom/Documents/data/rnn/mnist")
from mnist_rnn_model import show_res
from collections import deque

def import_data(file, rel_vars):
    data = pd.read_csv(file)
    data = data[rel_vars]
    return data

def cut_in_pieces(x, y):
    x_full = []
    x_item = deque(maxlen = n_time)
    y_full = []
    for idx, line in enumerate(x):
        x_item.append(line)
        if idx >= n_time-1:
            x_full.append(x_item)
            y_full.append(y[idx])
    x_new = np.array(x_full)
    y_new = np.array(y_full)
    return x_new, y_new

def build_network(n_time, input_dim: int, output_dim: int, n_hid1: int, n_hid2: int, learning_rate: float = 0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape = (n_time, input_dim)))
#    model.add(tf.keras.layers.SimpleRNN(n_hid1, return_sequences = False, activation = "tanh"))
    model.add(tf.keras.layers.LSTM(n_hid1, return_sequences = False, activation = "tanh"))
    model.add(tf.keras.layers.Dense(output_dim, activation = "softmax"))
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss, metrics = ["accuracy"])	
    return model

def preprocess_rain(data, data_size, test_perc, dep_var):
    data.replace("No", 0, inplace = True)
    data.replace("Yes", 1, inplace = True)
    data = data.dropna() # not perfectly correct, fix later
    y = np.array(data[dep_var])
    data.drop(columns = [dep_var], inplace = True)
    x = np.array(data)
    n_labels = int(np.max(y)+1)
    var_size = x.shape[1]
    x, y = x[:data_size], y[:data_size]
    x_new, y_new = cut_in_pieces(x, y)
    # split in train/test
    idx = np.arange(x_new.shape[0])
    np.random.shuffle(idx)
    cutoff = int(((1-test_perc)/100)*x_new.shape[0])
    train, test = idx[:cutoff], idx[cutoff:]
    print(cutoff, train, test)
    x_train, y_train, x_test, y_test = x_new[train], y_new[train], x_new[test], y_new[test]
    y_train = tf.one_hot(y_train, n_labels)
    y_test  = tf.one_hot(y_test, n_labels)
    return var_size, n_labels, x_train, y_train, x_test, y_test

def train_model():
    res = model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))
    return res

if __name__ == "__main__":
    # main program
    file = "weatherAUS.csv"
    rel_vars = ["MinTemp", "MaxTemp", "Humidity9am", "Humidity3pm",
			    "Rainfall", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow"]
    data = import_data(file, rel_vars)   

    # pre-processing
    dep_var = "RainTomorrow"
    n_time = 10
    data_size = 1000 # downscale to make data set smaller (and training faster)
    test_perc = 5    # percentage test stimuli
    var_size, n_labels, X_train, y_train, X_test, y_test = preprocess_rain(data, data_size, test_perc, dep_var)

    model = build_network(n_time, input_dim = var_size, output_dim = n_labels, n_hid1 = 20, n_hid2 = 10)
    res = train_model()
    show_res(res)
