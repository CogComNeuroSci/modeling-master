#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:25:11 2023
confabulate characters based on shakespeare oeuvre
(or any other text you want)
using a network with two (GRU) recurrent layers
@author: tom verguts
inspired by similar code by Cedric De Boom and Tim Verbelen
note that processed data must be stored as well bcs set is unordered
put train_it and save_it to False if you want to test an existing model
Can you predict what is different if you sample deterministically instead of 
probabilistically? Try it afterwards.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from os.path import join
import pickle

def build_network(batch_size: int, input_dim: int, output_dim: int, n_hid: int, learning_rate: float = 0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(batch_input_shape = (batch_size, stim_depth, input_dim)))
    model.add(tf.keras.layers.GRU(n_hid, return_sequences = True, stateful = True, activation = "tanh"))
    model.add(tf.keras.layers.GRU(n_hid, return_sequences = True, stateful = True, activation = "tanh"))
    model.add(tf.keras.layers.Dense(output_dim, activation = "softmax"))
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss, metrics = ["accuracy"])	
    return model

def train_model(n_times: int = 2, test_it: bool = False):
    for loop in range(n_times):
        print(f"training iteration {loop}")
        X_train, Y_train = \
         make_data(data, data_size, stim_depth, stim_dim)
        res = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 10, verbose = False)
        if test_it:
            test_model(n_cont = 100)
    return res

def test_model(n_cont = 50):
    n_seed = stim_depth
    x = np.zeros((batch_size, stim_depth, stim_dim))
    r = np.random.randint(0, len(data)- n_seed)
    seed = data[r:r+n_seed]
    seed_char = ''.join([itos[tok] for tok in seed])
    for loop in range(n_seed):
        x[0, loop, seed[loop]] = 1
    for row in range(n_seed-1, n_seed + n_cont):
        out = model(x)
        batch_nr, stim_nr = row//stim_depth, row%stim_depth
        probs = np.ndarray.flatten(out[batch_nr, stim_nr, :].numpy())
        y = np.random.choice(np.arange(len(chars)), p = probs)
#        y = np.argmax(probs)		# sample deterministically
        seed_char += itos[y]
        batch_nr, stim_nr = (row+1)//stim_depth, (row+1)%stim_depth
        x[batch_nr, stim_nr, y] = 1        
    print(seed_char[n_seed:], "\n")

def text2vec(text_file, verbose = False):
    """preprocessing of the data"""
    chars = set()
    data_length = 0
    banned_chars = set(['}', '<', '\\ufeff', '$'])
    with open(text_file, 'r') as infile:
      for line in infile:
        for char in line:
          if char not in banned_chars:
            chars.add(char)
            data_length += 1
    if verbose:
        print(f'Length of dataset: {data_length} chars')
        print(f'No. of unique chars: {len(chars)}')
    stoi = {c:i for i, c in enumerate(chars)}
    itos = {i:c for c, i in stoi.items()}
    data = np.zeros(data_length, dtype=np.int)
    i = 0
    with open(text_file, 'r') as infile:
      for line in infile:
        for char in line:
          if char not in banned_chars:
            data[i] = stoi[char]
            i += 1
    return data, chars, stoi, itos

def make_data(data, n_stim, stim_depth, stim_dim):
   X = np.zeros((n_stim, stim_depth, stim_dim))
   Y = np.zeros((n_stim, stim_depth, stim_dim))
   for loop in range(n_stim):
        k = np.random.randint(0, len(data)-stim_depth)
        for small_loop in range(stim_depth):
            X[loop, small_loop, data[k + small_loop]] = 1      
            Y[loop, small_loop, data[k + small_loop + 1]] = 1
   return X, Y

# start main code here
text = "shakespeare.txt"
train_it, save_it, model_nr = False, False, 1


if train_it:
    batch_size = 128
    data_size  = batch_size*20 # data for one model.fit()
    stim_depth = 100
    data, chars, stoi, itos = text2vec(text)
    stim_dim = len(chars)
    model = build_network(batch_size = batch_size, input_dim = stim_dim, output_dim = stim_dim, n_hid = 128)
    print("pre training:")
    test_model(n_cont = 20)
    res = train_model(n_times = 50, test_it = True)
    plt.plot(res.history["loss"])
else: # load model + processed data
    savedir = join(os.getcwd(), "models")
    model = tf.keras.models.load_model(
	  join(savedir,"model_shake"+str(model_nr)+".keras"))
    batch_size = model.input.shape[0]
    stim_depth = model.input.shape[1]
    with open(join(savedir, "texts"+str(model_nr)+".pkl"), 'rb') as f:  
        data, chars, stoi, itos = pickle.load(f)
    stim_dim = len(chars)
		
if save_it:
    savedir = join(os.getcwd(), "models")
    tf.keras.models.save_model(model, os.path.join(savedir,"model_shake"+str(model_nr)+".keras"), save_format = "keras")
    with open(join(savedir, "texts"+str(model_nr)+".pkl"), 'wb') as f:  
        pickle.dump([data, chars, stoi, itos], f)

print("post training:")
test_model(n_cont = 100)
