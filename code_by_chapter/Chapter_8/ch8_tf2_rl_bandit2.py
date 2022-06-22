#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tom verguts
implements a multi-armed bandit task
does estimation of weights for RL purposes in TF2
there is a single output unit; choices are encoded in the weights
note that action is separate from estimation; only the estimation part is thus optimal
if you want temperature parameter, easiest solution seems to be to multiply training data with it
in contrast to ch_8_RL_bandit, this code also acts (epsilon-greedy)
this approach combines aspects from ch 8 and 9 
"""

#%% imports and initializations
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n_trials = 500
learning_rate = 0.1
epsilon = 0.1
buffer_size = 10
p = np.array([0.2, 0.2, 0.2, 0.95])  # payoff probabilities
n_action = p.size
actions = np.arange(n_action)
action_1hot = np.eye(n_action)
optimal_action = np.argmax(p)
x_data = np.zeros((buffer_size, n_action))
y_data = np.zeros((buffer_size, 1))
reward = np.zeros(n_trials)
optimal = np.zeros(n_trials)

#%% model construction
model = tf.keras.Sequential([
			tf.keras.Input(shape=(n_action,)),
			tf.keras.layers.Dense(1, activation = "sigmoid")
			] )
model.build()
loss = tf.keras.losses.BinaryCrossentropy()
#loss = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = opt, loss = loss)

def learn(x_data, y_data):
	x_train, y_train = x_data, y_data
	model.fit(x_train, y_train, batch_size = 1, epochs = 1)	

def update_buffer(x_data, y_data, action, reward):
	x_data[0:buffer_size-1,:] = x_data[1:buffer_size,:]
	y_data[0:buffer_size-1,:] = y_data[1:buffer_size,:]
	x_data[-1] = action_1hot[action,:]
	y_data[-1] = reward
	return x_data, y_data	

#%% main code
for loop in range(n_trials):
	# sample a bandit
	if np.random.uniform()<epsilon: # explore
		action = np.random.choice(actions)
	else:                           # exploit 
		action = np.argmax(np.array(model.layers[0].weights[0]))
	reward[loop] = (np.random.uniform()<p[action])*1
	optimal[loop] = (action == optimal_action)*1
	x_data, y_data = update_buffer(x_data, y_data, action, reward[loop])
	# learn	
	if (loop%buffer_size == 0) and (loop > 0):
		learn(x_data, y_data)		

#%% show results
filter_size = 5
filt= np.ones(filter_size)/filter_size
fig, axs = plt.subplots(nrows = 1, ncols = 2)
axs[0].plot(np.convolve(reward, filt)[filter_size:-filter_size], color = "black")
axs[1].plot(np.convolve(optimal, filt)[filter_size:-filter_size], color = "black")

# print weights
#print("model weights:")
#print(model.layers[0].weights)