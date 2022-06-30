#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
solves the cart pole problem with deep q learning and episode replay
as described in mnih et al
i was inspired by machine learning with phil for this implementation
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def build_network(input_dim: int, action_dim: int, learning_rate, hidlayer1: int, hidlayer2: int):
    model = tf.keras.Sequential([ 
            tf.keras.layers.Dense(hidlayer1, input_shape = (input_dim,), activation = "relu", name = "layer0"),
            tf.keras.layers.Dense(hidlayer2, activation = "relu", name = "layer1"),
            tf.keras.layers.Dense(action_dim, activation = "linear", name = "layer2")
			] )
    model.build()
    loss = {"layer2": tf.keras.losses.MeanSquaredError()}
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss)
    return model



class Agent(object):
    def __init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, epsilon_dec, lr, gamma, learn_gran, nhid1, nhid2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions = np.arange(n_actions)
        self.buffer_size = buffer_size
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_dec = epsilon_dec
        self.epsilon     = epsilon_max
        self.lr = lr
        self.gamma = gamma
        self.learn_gran = learn_gran
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.network = build_network(self.n_states, self.n_actions, self.lr, self.nhid1, self.nhid2)
        self.x_buffer = np.zeros((self.buffer_size, self.n_states))
        self.xn_buffer = np.zeros((self.buffer_size, self.n_states))
        self.y_buffer = np.zeros((self.buffer_size, self.n_actions))
        self.r_buffer = np.zeros((self.buffer_size, 1))
        self.d_buffer = np.zeros((self.buffer_size, 1))


    def update_buffer(self, data, n_step, location):        
        for data_loop in range(n_step):
            self.x_buffer[(location + data_loop)%self.buffer_size]  = data[data_loop, 0:self.n_states]
            self.xn_buffer[(location + data_loop)%self.buffer_size] = data[data_loop, self.n_states:2*self.n_states]
            action_1hot = np.zeros(self.n_actions)
            action_1hot[data[data_loop, -3].astype(int)] = 1
            self.y_buffer[(location + data_loop)%self.buffer_size] = action_1hot
            self.r_buffer[(location + data_loop)%self.buffer_size] = data[data_loop, -2].astype(int)
            self.d_buffer[(location + data_loop)%self.buffer_size] = data[data_loop, -1]
        return np.minimum(location + n_step, self.buffer_size)                    
            
    def learn(self, n: int, verbose: bool = True):
        #self.epsilon = self.epsilon_max # in case you want to reset epsilon on each episode
        sample_size = np.minimum(100, n);
        sample = np.random.choice(n, sample_size)
        q_predict = self.network.predict(self.x_buffer[sample])
        q_next = self.network.predict(self.xn_buffer[sample])
        q_max = np.amax(q_next, axis = 1)
        include_v = 1 - self.d_buffer[sample]
        if verbose:
            print("x buffer:", self.x_buffer)
            print("n: ", n)
            print("q_predict", q_predict)
            print("q_next", q_next)
        q_target = q_predict.copy()
        target_indices = np.dot(self.y_buffer[sample], np.arange(self.n_actions)).astype(int)
#        print(target_indices)
        q_target[list(range(q_target.shape[0])), target_indices] = np.squeeze(self.r_buffer[sample])
        q_target[list(range(q_target.shape[0])), target_indices] += self.gamma*q_max * np.squeeze(include_v)
       	self.network.fit(self.x_buffer[sample], q_target, batch_size = 64, epochs = 2000, verbose = 0)	
        if verbose:
            print("q_target", q_target)
           
    def sample(self, state):
        if np.random.uniform() < self.epsilon:
           action = np.random.choice(self.actions)
        else:
            y = self.network.predict(np.array(state[np.newaxis,:]))
            action = np.argmax(y)
        self.epsilon = np.max([self.epsilon_min, self.epsilon*self.epsilon_dec]) 
        return action


def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4):
    lc = np.zeros(n_loop)
    buffer_count = 0
    stop_crit = False
    loop, success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
        data = np.zeros((max_n_step, input_dim*2 + 3)) # data for this loop
        while not done:
            action = rl_agent.sample(state)
            next_state, reward, done, info = env.step(action)
            data[n_step, 0:input_dim] = state
            data[n_step, input_dim:2*input_dim] = next_state
            data[n_step, -3]  = action
            if not done or n_step >= max_n_step-1:
                data[n_step, -2]  = reward
            else:
                data[n_step, -2]  = -100
            data[n_step, -1] = int(done)
            n_step += 1
            state = next_state
        buffer_count = rl_agent.update_buffer(data, n_step, buffer_count)
        if (not loop % rl_agent.learn_gran) and (buffer_count > 500): # don't learn first 500 trials
            rl_agent.learn(buffer_count, verbose = False)
        lc[loop] = n_step
        loop += 1
        success += (n_step == max_n_step)
        print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (success > 10)
    return lc, success > 10

def perform(env, rl_agent, verbose: bool = False):
    state = env.reset()
    n_step, done = 0, False
    while not done:
        env.render()
        action = rl_agent.sample(state)
        next_state, reward, done, info = env.step(action)
        n_step += 1
        state = next_state
        if verbose:
            print(n_step)
        
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    load_model, save_model, train_model = False, False, True
    rl_agent = Agent(env.observation_space.shape[0], env.action_space.n, \
                           buffer_size = 1000, epsilon_min = 0.001, epsilon_max = 0.99, \
                           epsilon_dec = 0.999, lr = 0.001, gamma = 0.9, learn_gran = 1, nhid1 = 8, nhid2 = 4)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/model_cartpole")
    if train_model:
        lc, solved = learn_w(env, n_loop = 50, max_n_step = 200, input_dim = env.observation_space.shape[0])
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/model_cartpole")
    if train_model:
        plt.plot(lc)
    if train_model and solved:
        print("Problem solved.")
    perform(env, rl_agent, verbose = False)
    env.close()