#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
under construction
"""

import gym, time
import tensorflow as tf

def build_network(input_dim, action_dim, learning_rate):
    model = tf.keras.Sequential([ 
			tf.keras.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(action_dim, activation = "sigmoid"),
			tf.keras.layers.Dense(1, activation = "sigmoid")
			] )
    model.build()
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))
    return model

#class Agent(object):
    def __init__(self, n_states, n_hid, n_actions, lr):
        self.n_states = n_states
        self.n_hid = n_hid
        self.n_actions = n_actions
        self.lr = lr
        self.network = build_network(env.observation_space.shape, env.action_space.shape[0], lr)
    def learn():
       
    def sample():
       
        
env = gym.make("CartPole-v0")
env.reset()

# explore
print(env.action_space)
print(env.observation_space)

# algo
for _ in range(10):
    env.render()
    action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation)
time.sleep(1)
env.close()