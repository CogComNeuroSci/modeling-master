#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:37:36 2020

@author: tom verguts
mixing two tasks;
applies backprop with both shared and unique units across the two tasks
illustrates Keras functional API
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from itertools import product

def build_model(n_in, n_shared, n_unique, n_out):
    input1  = Input((n_in,), name = "input1") 
    input2  = Input((n_in,), name = "input2") 
    inputs  = Concatenate()([input1, input2])
    shared  = Dense(n_shared, activation = "sigmoid")(inputs)
    unique1 = Dense(n_unique, activation = "sigmoid")(input1)
    unique2 = Dense(n_unique, activation = "sigmoid")(input2)
    hid     = Concatenate()([shared, unique1, unique2])
    output  = Dense(n_out, activation = "softmax")(hid)
    model   = Model([input1, input2], output)
    return model

def task2(v, overlap):
    # define task 2 mapping
    n_diff = int((1-overlap)*train_x.shape[0])
    ix = np.random.permutation(range(train_x.shape[0]))
    w = np.copy(v)
    w[ix[:n_diff]] = 1 - w[ix[:n_diff]] 
    return w

def array_to_dict(x):
    return {"input1": x[:,:stim_dim], "input2": x[:,stim_dim:]}
	
def step(model, X, y):
	# keep track of our gradients
    with tf.GradientTape() as tape:
     # make a prediction using the model and then calculate the loss
        dict_X = array_to_dict(X)
        pred = model(inputs = dict_X) 
        loss = categorical_crossentropy(y, pred)
	    # calculate the gradients using our tape and then update the model weights
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

def extend(idx, x):
    # write the stimuli in the right "bank"
    vec = [1 - idx, idx]
    x_extended = np.kron(vec, x) 
    return x_extended
	
task_overlap, unit_overlap = 1, .0
learning_rate = 0.1
epochs = 1000
n_hid = 4
n_shared = int(n_hid*unit_overlap)
n_unique = int((n_hid - n_shared)/2)
stim_dim = 3
train_x = []
for row in product([0, 1], repeat = stim_dim):
    train_x.append(row)
train_x = np.array(train_x)
n = int(train_x.shape[0]/2)

n_sim = 2
total_acc_overall = 0

for sim_loop in range(n_sim):
    task1 = np.array([0]*n + [1]*n)
    np.random.shuffle(task1)
    train_t = np.array([task1, task2(task1, overlap = task_overlap)])
    train_t = np.transpose(train_t)
    train_t = np.array([to_categorical(train_t[:, 0], 2), to_categorical(train_t[:, 1], 2)]) 
    error_function = np.zeros(epochs)
    opt  = tf.keras.optimizers.SGD(learning_rate = learning_rate)

    model = build_model(n_in = stim_dim, n_shared = n_shared, n_unique = n_unique, n_out = 2)
    model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=["acc"])

    for epoch in range(epochs):
        task_idx = int(np.random.randn()>0)
        ix = np.random.permutation(range(train_x.shape[0]))
        x_shuffle = train_x[ix]
        t_shuffle = train_t[task_idx]
        t_shuffle = t_shuffle[ix]
        x_shuffle = extend(task_idx, x_shuffle)
        step(model, x_shuffle, t_shuffle)
        dict_x = array_to_dict(x_shuffle)
#        (error_function[epoch], acc) = model.evaluate(dict_x, t_shuffle, verbose = 0)

    total_acc = 0
    plt.plot(range(epochs), error_function)
    print("\n")
    for indx in range(2):
        x = extend(indx, train_x)
        dict_x = array_to_dict(x)
        y = train_t[indx]
        (loss, acc) = model.evaluate(dict_x, y, verbose = 0)
        total_acc  += acc
        print(f"test accuracy task {indx}: {acc:.2f}")

    total_acc = total_acc/(indx+1)
    print(f"test accuracy for both tasks: {total_acc:.2f}")
    total_acc_overall += total_acc

total_acc_overall /= n_sim
print(f"test accuracy over sims {total_acc_overall:.2f}")	
	