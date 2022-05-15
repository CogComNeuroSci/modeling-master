#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
under construction
"""

import gym, time
import tensorflow as tf
import numpy as np

def build_network(input_dim, action_dim, learning_rate):
    model = tf.keras.Sequential([ 
			tf.keras.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(action_dim, activation = "sigmoid")
			] )
    model.build()
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss)
    return model

class Agent(object):
    def __init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, epsilon_dec, lr, learn_gran):
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions = np.arange(n_actions)
        self.buffer_size = buffer_size
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_dec = epsilon_dec
        self.lr = lr
        self.learn_gran = learn_gran
        self.network = build_network(self.n_states, self.n_actions, self.lr)
        self.x_buffer = np.zeros((self.buffer_size, self.n_states))
        self.y_buffer = np.zeros((self.buffer_size, self.n_actions))
        self.r_buffer = np.zeros((self.buffer_size, 1))

    def update_buffer(self, observation, action, reward):
        self.x_buffer[0:self.buffer_size-1,:] = self.x_buffer[1:self.buffer_size,:]
        self.y_buffer[0:self.buffer_size-1,:] = self.y_buffer[1:self.buffer_size,:]
        self.r_buffer[0:self.buffer_size-1,:] = self.r_buffer[1:self.buffer_size,:]
        action_1hot = np.zeros(self.n_actions)
        action_1hot[action] = 1
        self.x_buffer[-1] = observation
        self.y_buffer[-1] = action_1hot
        self.r_buffer[-1] = reward
        
    def learn(self):
        q_predict = self.network.predict(self.x_buffer)
        q_target = q_predict.copy()
        target_indices = np.dot(self.y_buffer, np.arange(self.n_actions)).astype(int)
        q_target[:, target_indices] = self.r_buffer
       	self.network.fit(self.x_buffer, q_target, batch_size = 1, epochs = 1)	
           
    def sample(self, observation):
        if np.random.uniform()< self.epsilon:
           action = np.random.choice(self.actions)
        else:
            y = self.network.predict(np.array(observation[np.newaxis,:]))
            action = np.argmax(y)
        self.epsilon = np.max([self.epsilon_min, self.epsilon*self.epsilon_dec]) 
        return action

env = gym.make("CartPole-v0")
env.reset()
my_rl_agent = Agent(env.observation_space.shape[0], env.action_space.n, \
                           buffer_size = 10, epsilon_min = 0.01, epsilon_max = 0.5, epsilon_dec = 0.8, lr = 0.1, learn_gran = 1)

# explore
print(env.action_space)
print(env.observation_space)

# algo
action = env.action_space.sample()
n_step, done = 0, False
while not done:
    env.render()
    observation, reward, done, info = env.step(action)
    action = my_rl_agent.sample(observation)
    my_rl_agent.update_buffer(observation, action, reward)
    if not n_step % my_rl_agent.learn_gran:
        my_rl_agent.learn()
    print(n_step)
    print(my_rl_agent.epsilon)
    n_step+= 1
    
time.sleep(1)
env.close()