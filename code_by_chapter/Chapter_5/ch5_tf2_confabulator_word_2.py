#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:25:11 2023
@author: tom verguts
- confabulate words based on shakespeare oeuvre
(or any other text you want)
- version 2 uses more efficient representations (embedding)
- using a network with two recurrent (GRU) layers
- inspired by similar code by Cedric De Boom and Tim Verbelen
- note that processed data must be stored as well bcs set is unordered
- put train_it and save_it to False if you want to test an existing model
- you can make the path deterministic (always the same) by setting the random seed;
- you can make sure you sample the same word in the same situation by sampling via argmax;
try it out and try to predict what will be different
- var max_line is useful to speed up the process (the word version is slow with all the text)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from os.path import join
import pickle

def build_network(batch_size: int, input_dim: int, output_dim: int, n_hid: int, learning_rate: float = 0.001):
    embedding_dim = 40
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = stim_dim, output_dim = embedding_dim, batch_size = batch_size, input_length = stim_depth))
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
    x = np.zeros((batch_size, stim_depth))
    r = np.random.randint(0, len(data)- n_seed)
    seed = data[r:r+n_seed]
    words_generated = [itos[tok] for tok in seed]
    for loop in range(n_seed):
        x[0, loop] = seed[loop]
    for row in range(n_seed-1, n_seed + n_cont):
        out = model(x)
        batch_nr, stim_nr = row//stim_depth, row%stim_depth
        probs = np.ndarray.flatten(out[batch_nr, stim_nr, :].numpy())
        y = np.random.choice(np.arange(len(words)), p = probs)
#        y = np.argmax(probs)		# sample the highest prob word
        words_generated.append(itos[y])
        batch_nr, stim_nr = (row+1)//stim_depth, (row+1)%stim_depth
        x[batch_nr, stim_nr] = y        
    print(" ".join(words_generated[n_seed:]), "\n")

def text2vec(text_file, verbose = False, start_line = None, max_line = None):
    """preprocessing of the data"""
    line_nr = 0
    include = False
    words = set()
    data_length = 0
    banned_chars = set(['}', '<', '\\ufeff', '$'])
    with open(text_file, 'r') as infile:
        for line in infile:
            if (line == start_line) or not start_line:
	            include = True 
            if include:
                line_nr += 1
                if line_nr == max_line:
                    break
                for word in line.split():
                    banned = False
                    for char in banned_chars:
                        if char in word:
                            banned = True
                            break
                    if not banned:
                        words.add(word)
                        data_length += 1
    if verbose:
        print(f'Length of dataset: {data_length} words')
        print(f'No. of unique words: {len(words)}')
    stoi = {c:i for i, c in enumerate(words)}
    itos = {i:c for c, i in stoi.items()}
    data = np.zeros(data_length, dtype=np.int)
    i = 0
    include = False
    line_nr = 0
    with open(text_file, 'r') as infile:
        for line in infile:
            if (line == start_line) or not start_line:
                include = True
            if include:
                line_nr += 1
                if line_nr == max_line:
                    break
                for word in line.split():
                    banned = False
                    for char in banned_chars:
                        if char in word:
                            banned = True
                            break
                    if not banned:
                        data[i] = stoi[word]
                        i += 1
    return data, words, stoi, itos

def make_data(data, n_stim, stim_depth, stim_dim):
   X = np.zeros((n_stim, stim_depth))
   Y = np.zeros((n_stim, stim_depth, stim_dim))
   for loop in range(n_stim):
        k = np.random.randint(0, len(data)-stim_depth)
        for small_loop in range(stim_depth):
            X[loop, small_loop] = data[k + small_loop]      
            Y[loop, small_loop, data[k + small_loop + 1]] = 1
   return X, Y

# start main code here
text = "shakespeare.txt"
train_it, save_it, model_nr = True, True, 1


if train_it: # train the model
    batch_size = 128           # how many stimuli (of length (stim_depth, stim_dim)) per batch
    data_size  = batch_size*20 # data for one model.fit()
    stim_depth = 10            # during training, how far into the past do you go to predict the next word
    data, words, stoi, itos = text2vec(text, verbose = True, start_line = "PLAYS\n", max_line = 1000)
    stim_dim = len(words)
    model = build_network(batch_size = batch_size, input_dim = stim_dim, output_dim = stim_dim, n_hid = 128)
    print("pre training:")
    test_model(n_cont = 20)
    res = train_model(n_times = 10, test_it = True) # the length of this training determines execution time
    plt.plot(res.history["loss"])
else: # load the model + processed data
    savedir = join(os.getcwd(), "models_word")
    model = tf.keras.models.load_model(
	  join(savedir,"model_shake"+str(model_nr)+".keras"))
    batch_size = model.input.shape[0]
    stim_depth = model.input.shape[1]
    with open(join(savedir, "texts"+str(model_nr)+".pkl"), 'rb') as f:  
        data, words, stoi, itos = pickle.load(f)
    stim_dim = len(words)
	
if save_it:
    savedir = join(os.getcwd(), "models_word")
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    tf.keras.models.save_model(model, os.path.join(savedir,"model_shake"+str(model_nr)+".keras"), save_format = "keras")
    with open(join(savedir, "texts"+str(model_nr)+".pkl"), 'wb') as f:  
        pickle.dump([data, words, stoi, itos], f)

print("post training:") # a walk in word space
#np.random.seed(2002)
test_model(n_cont = 100)
