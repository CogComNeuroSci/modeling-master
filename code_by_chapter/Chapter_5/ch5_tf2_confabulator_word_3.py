#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:25:11 2023
@author: tom verguts
- confabulate words based on some text you want
- version 3 uses multihead attention (as in a transformer)
- see version 1 for other info
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import os, pickle, sys
from os.path import join
sys.path.append("Users/tom/Documents/Modcogproc/modeling-master/code_by_chapter/Chapter_5/")
from ch5_tf2_confabulator_word import text2vec, make_data

def build_network(batch_size: int, input_dim: int, output_dim: int, n_hid: int, learning_rate: float = 0.001):
    input1    = tf.keras.Input(batch_input_shape = (batch_size, stim_depth, input_dim))
    attention = tf.keras.layers.MultiHeadAttention(num_heads = 2, key_dim = 4, value_dim = 4)(input1, input1)
    output    = tf.keras.layers.Dense(output_dim, activation = "softmax")(attention)
    loss      = tf.keras.losses.CategoricalCrossentropy()
    model     = Model(input1, output)
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
    words_generated = [itos[tok] for tok in seed]
    for loop in range(n_seed):
        x[0, loop, seed[loop]] = 1
    for row in range(n_seed-1, n_seed + n_cont):
        out = model(x)
        batch_nr, stim_nr = row//stim_depth, row%stim_depth
        probs = np.ndarray.flatten(out[batch_nr, stim_nr, :].numpy())
        y = np.random.choice(np.arange(len(words)), p = probs)
#        y = np.argmax(probs)		# sample the highest prob word
        words_generated.append(itos[y])
        batch_nr, stim_nr = (row+1)//stim_depth, (row+1)%stim_depth
        x[batch_nr, stim_nr, y] = 1        
    print(" ".join(words_generated[n_seed:]), "\n")

# start main code here
text = "shakespeare.txt"
train_it, save_it, model_nr = True, True, 1

if train_it: # train the model
    batch_size = 128           # how many stimuli (of length (stim_depth, stim_dim)) per batch
    data_size  = batch_size*20 # data for one model.fit()
    stim_depth = 10            # during training, how far into the past do you go to predict the next word
    data, words, stoi, itos = text2vec(text, verbose = True, start_line = "PLAYS\n", max_line = 100)
    stim_dim = len(words)
    model = build_network(batch_size = batch_size, input_dim = stim_dim, output_dim = stim_dim, n_hid = 128)
    print("pre training:")
    test_model(n_cont = 20)
    res = train_model(n_times = 5, test_it = True) # the length of this training determines execution time
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
