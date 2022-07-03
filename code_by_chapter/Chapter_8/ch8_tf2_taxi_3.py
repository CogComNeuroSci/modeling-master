#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
solves the taxi problem with deep q learning and episode replay
as described in mnih et al
reduced input space for hierarchical processing? thus far not very spectacular
"""

#%% imports and definitions
import gym
import tensorflow as tf
import numpy as np
import os
from ch8_tf2_taxi_2 import Agent, perform, learn_w, plot_data

def build_network(input_dim: int, action_dim: int, learning_rate: float, nhid1: int, nhid2: int):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(nhid1, input_shape = (input_dim,), activation = "relu"))
    if nhid2 > 0:
        model.add(tf.keras.layers.Dense(nhid2, activation = "relu"))
    model.add(tf.keras.layers.Dense(action_dim, activation = "linear", name = "outputlayer"))
    model.build()
    loss = {"outputlayer": tf.keras.losses.MeanSquaredError()}
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss)
    return model

def decode(i):
    # taken from gym github bcs could not be used in current env class
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return list(reversed(out))

class Agent_hier(Agent):
    """ just like the Agent class from taxi_2, the buffers store info 1-dimensional;
    but the input space is lower-dimensional, and there are hidden units to explot (hierarchical?) structure
    """
    def __init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, 
                             epsilon_dec, lr, gamma, learn_gran, update_gran, nhid1, nhid2):
        Agent.__init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, 
                              epsilon_dec, lr, gamma, learn_gran, update_gran)
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.network = build_network(self.n_states, self.n_actions, self.lr, self.nhid1, self.nhid2)
        self.network_target = build_network(self.n_states, self.n_actions, self.lr, self.nhid1, self.nhid2)
    
    def fillup(self, state_list):
        v = np.zeros(self.n_states)
        v[state_list[0]]       = 1
        v[5 + state_list[1]]   = 1
        v[2*5 + state_list[2]] = 1
        v[3*5 + state_list[3]] = 1
        assert np.sum(v) == 4
        return v

    def learn(self, n: int, verbose: bool = True):
        #self.epsilon = self.epsilon_max # in case you want to reset epsilon on each episode
        sample_size = np.minimum(200, n)
        sample = np.random.choice(n, sample_size)
        v_x = np.zeros((sample_size, self.n_states))
        for idx, state in enumerate(self.x_buffer[sample]):
            state_list = decode(state)
            v = self.fillup(state_list)
            v_x[idx] = v
        q_predict = self.network.predict(v_x)
        v_xn = np.zeros((sample_size, self.n_states))
        for idx, state in enumerate(self.xn_buffer[sample]):
            state_list = decode(state)
            v = self.fillup(state_list)
            v_x[idx] = v
        q_next = self.network_target.predict(v_xn)
        q_max = np.amax(q_next, axis = 1)
        include_v = 1 - self.d_buffer[sample]
        q_target = q_predict.copy()
        target_indices = np.squeeze(np.array(self.y_buffer[sample]))
        q_target[list(range(q_target.shape[0])), target_indices] = np.squeeze(self.r_buffer[sample])
        q_target[list(range(q_target.shape[0])), target_indices] += self.gamma*q_max * np.squeeze(include_v)
       	self.network.train_on_batch(v_x, q_target)	
        if verbose:
            print("x buffer:", self.x_buffer)
            print("q_predict", q_predict)
            print("q_next", q_next)
            print("q_target", q_target) 

    def sample_soft(self, state):
        ## softmax sampling
        v = np.zeros((1, self.n_states))
        state_list = decode(state)
        v = self.fillup(state_list)
        v = v[np.newaxis, :]
        y = self.network.predict(np.array(v))
        prob = np.exp(y)
        prob = np.squeeze(prob/np.sum(prob))
        action = np.random.choice(range(env.action_space.n), p = prob)
        return action 
        

#%% 
if __name__ == "__main__":
    env = gym.make("Taxi-v2")
    load_model, save_model, train_model, performance, plot_results = False, True, True, False, True
    n_states = 25 + 5 + 4 # from multiplication to addition!
    rl_agent = Agent_hier(n_states, env.action_space.n, \
                           buffer_size = 200, epsilon_min = 0.1, epsilon_max = 1, \
                           epsilon_dec = 0.99, lr = 0.001, gamma = 0.99, learn_gran = 1, \
                           update_gran = 5, nhid1 = 32, nhid2 = 16)
    if load_model:
        rl_agent.network_target = tf.keras.models.load_model(os.getcwd()+"/model_taxi_dqn.h5")
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/model_taxi_dqn.h5")
    if train_model:
        lc, solved, reward_vec = learn_w(env, rl_agent, n_loop = 500, 
                                        max_n_step = 200, success_crit = 200)
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/model_taxi_dqn.h5")
    if plot_results:
        plot_data(window = 10, reward_vec = reward_vec, lc = lc)
    if train_model and solved:
        print("Problem solved.")
    if performance:
        perform(env, rl_agent, verbose = True)
    env.close()