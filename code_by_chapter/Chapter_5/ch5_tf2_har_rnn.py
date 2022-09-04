#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 15:16:21 2022

@author: tom verguts
try out recurrent models on posture data; 
specifically the Human Activity Recognition dataset on kaggle
"""
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/tom/Documents/data/rnn/mnist") # or wherever you store mnist_rnn_model
from mnist_rnn_model import show_res
from collections import deque

def read_data(data_dir: str):
    X_train = np.loadtxt(os.path.join(data_dir, "train", "X_train.txt")).astype(np.float16)
    y_train = np.loadtxt(os.path.join(data_dir, "train", "y_train.txt")).astype(np.float16)
    X_test =  np.loadtxt(os.path.join(data_dir, "test", "X_test.txt")).astype(np.int)
    y_test = np.loadtxt(os.path.join(data_dir, "test", "y_test.txt")).astype(np.int)
    return X_train, y_train, X_test, y_test

def cut_in_pieces(x, y):
    x_full = []
    x_item = []
    y_full = []
    for idx, line in enumerate(x):
        x_item.append(line)
        if not ((idx+1) % n_time):
            if np.all(y[idx:idx-n_time] == y[idx]): # throw away mixed-label batch
                x_full.append(x_item)
                y_full.append(y[idx])
            x_item = []
    x_new = np.array(x_full)
    y_new = np.array(y_full)
    return x_new, y_new

def cut_in_pieces2(x, y):
	# better way to preprocess the data: keeping more stimuli
    x_full = []
    x_item = deque(maxlen = n_time)
    y_full = []
    for idx, line in enumerate(x):
        x_item.append(line)
        if idx >= n_time-1:
            if np.all(y[idx:idx-n_time] == y[idx]): # throw away mixed-label batch
                x_full.append(x_item)
                y_full.append(y[idx])
    x_new = np.array(x_full)
    y_new = np.array(y_full)
    return x_new, y_new

def preprocess_pose(x_train, y_train, train_size,
					x_test, y_test, test_size):
    n_labels = int(np.max(y_train)+1)
    stim_size = x_train.shape[1]
    x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]
    x_train_preproc, y_train_preproc = cut_in_pieces2(x_train, y_train)
    x_test_preproc, y_test_preproc   = cut_in_pieces2(x_test, y_test)
    y_train = tf.one_hot(y_train_preproc, n_labels)
    y_test  = tf.one_hot(y_test_preproc, n_labels)
    return stim_size, n_labels, x_train_preproc, y_train, x_test_preproc, y_test

def build_network(image_size: int, output_dim: int, n_hid1: int, n_hid2: int, learning_rate: float = 1e-5):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape = (n_time, image_size)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LSTM(n_hid1, return_sequences = True, activation = "tanh"))
    model.add(tf.keras.layers.LSTM(n_hid2, return_sequences = False, activation = "tanh"))
    model.add(tf.keras.layers.Dense(output_dim, activation = "sigmoid", name = "outputlayer"))
    loss = {"outputlayer": tf.keras.losses.MeanSquaredError()}
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate, decay = 1e-6), loss = loss, metrics = ["accuracy"])	
    return model	

def train_model(epochs):
    res = model.fit(X_train, y_train, batch_size = 1, epochs = epochs, validation_data = (X_test, y_test) )
    return res

def test_model(X, y):
    y_pred = model.predict(X)
    y_pred_label= np.argmax(y_pred, axis = 1) + 1
    return np.mean(y_pred_label == y)


# main program
X_train, y_train, X_test, y_test = read_data("UCI HAR Dataset")
n_time = 10 # how many time steps do we use to predict the movement
train_size, test_size = X_train.shape[0], X_test.shape[0] # can downscale to make data set smaller (and training faster)
#train_size, test_size = 40, 10 # can downscale to make data set smaller (and training faster)
image_size, n_labels, X_train, y_train, X_test, y_test = preprocess_pose(
                                  X_train, y_train, train_size, X_test, y_test, test_size)
print(X_train.shape)
model = build_network(image_size = image_size, output_dim = n_labels, n_hid1 = 50, n_hid2 = 10)
res = train_model(epochs = 20)
show_res(res)

