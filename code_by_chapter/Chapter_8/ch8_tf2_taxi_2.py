#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
solves the taxi problem with deep q learning and episode replay
as described in mnih et al
build_network0 has zero hidden layers... should be similar to tabular version
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append('/Users/tom/Documents/Modcogproc/modeling-master/code_by_chapter/Chapter_9')
from ch9_RL_taxi import smoothen

def build_network0(input_dim: int, action_dim: int, learning_rate: float):
##model without hidden layers
    initializer = tf.keras.initializers.Constant(1)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(action_dim, activation = "linear", kernel_initializer=initializer, name = "layer"))
    model.build()
    loss = {"layer": tf.keras.losses.MeanSquaredError()}
    model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss)
    return model


class Agent(object):
## similar to the earlier Agent classes but importing the older wasn't worth it in this case    
    def __init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, epsilon_dec, lr, gamma, learn_gran):
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
        self.network = build_network0(self.n_states, self.n_actions, self.lr)
        self.x_buffer = np.zeros((self.buffer_size, 1)).astype(int)
        self.xn_buffer = np.zeros((self.buffer_size, 1)).astype(int)
        self.y_buffer = np.zeros((self.buffer_size, 1)).astype(int)
        self.r_buffer = np.zeros((self.buffer_size, 1))
        self.d_buffer = np.zeros((self.buffer_size, 1))


    def update_buffer(self, data, n_step, location):        
        for data_loop in range(n_step):
            self.x_buffer[(location + data_loop)%self.buffer_size]  = data[data_loop, 0]
            self.xn_buffer[(location + data_loop)%self.buffer_size] = data[data_loop, 1]
            self.y_buffer[(location + data_loop)%self.buffer_size] =  data[data_loop, 2]
            self.r_buffer[(location + data_loop)%self.buffer_size] = data[data_loop,  3]
            self.d_buffer[(location + data_loop)%self.buffer_size] = data[data_loop,  4]
        return np.minimum(location + n_step, self.buffer_size)                    
            
    def learn(self, n: int, verbose: bool = True):
        #self.epsilon = self.epsilon_max # in case you want to reset epsilon on each episode
        sample_size = np.minimum(100, n)
        sample = np.random.choice(n, sample_size)
        v_x = np.zeros((sample_size, self.n_states))
        v_x[list(range(sample_size)), np.squeeze(self.x_buffer[sample])] = 1
        q_predict = self.network.predict(v_x)
        v_xn = np.zeros((sample_size, self.n_states))
        v_xn[list(range(sample_size)), np.squeeze(self.xn_buffer[sample])] = 1
        q_next = self.network.predict(v_xn)
        q_max = np.amax(q_next, axis = 1)
        include_v = 1 - self.d_buffer[sample]
        if verbose:
            print("x buffer:", self.x_buffer)
            print("n: ", n)
            print("q_predict", q_predict)
            print("q_next", q_next)
        q_target = q_predict.copy()
        target_indices = np.squeeze(np.array(self.y_buffer[sample]))
        q_target[list(range(q_target.shape[0])), target_indices] = np.squeeze(self.r_buffer[sample])
        q_target[list(range(q_target.shape[0])), target_indices] += self.gamma*q_max * np.squeeze(include_v)
       	self.network.train_on_batch(v_x, q_target)	
        if verbose:
            print("q_target", q_target)
        
    def sample(self, state):
    ## epsilon-greedy sample    
        if np.random.uniform() < self.epsilon:
           action = np.random.choice(self.actions)
        else:
           v = np.zeros((1, self.n_states))
           v[0, state] = 1
           y = self.network.predict(np.array(v))
           action = np.argmax(y)
        self.epsilon = np.max([self.epsilon_min, self.epsilon*self.epsilon_dec]) 
        return action

    def sample_soft(self, state):
    ## softmax sampling
        v = np.zeros((1, self.n_states))
        v[0, state] = 1
        y = self.network.predict(np.array(v))
        prob = np.exp(y)
        prob = np.squeeze(prob/np.sum(prob))
        action = np.random.choice(range(env.action_space.n), p = prob)
        return action

    
def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4):
    W= np.zeros((5, 500, 6))
    lc = np.zeros(n_loop)
    reward_vec = np.zeros(n_loop)
    buffer_count = 0
    stop_crit = False
    loop, success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
        data = np.zeros((max_n_step, 5)) # data for this loop
        while (not done) and (n_step < max_n_step):
            action = rl_agent.sample_soft(state)
            next_state, reward, done, info = env.step(action)
            data[n_step, 0] = state
            data[n_step, 1] = next_state
            data[n_step, 2]  = action
            data[n_step, 3]  = reward
            data[n_step, 4] = int(done)
            n_step += 1
            state = next_state
            reward_vec[loop] += reward
        buffer_count = rl_agent.update_buffer(data, n_step-1, buffer_count)
        if (not loop % rl_agent.learn_gran) and (buffer_count > 10): # don't learn first n trials
            rl_agent.learn(buffer_count, verbose = False)
        lc[loop] = n_step
        reward_vec[loop] /= n_step    
        loop += 1
        success += (reward == 20)
        print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (success > 100)

    return lc, success > 100, reward_vec, W

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
    env = gym.make("Taxi-v2")
    load_model, save_model, train_model, performance, plot_results = False, False, False, False, False
    rl_agent = Agent(env.observation_space.n, env.action_space.n, \
                           buffer_size = 100, epsilon_min = 0.1, epsilon_max = 1, \
                           epsilon_dec = 0.99, lr = 0.07, gamma = 0.95, learn_gran = 1)

    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/model_taxi_dqn.h5")
    if train_model:
        lc, solved, reward_vec, W = learn_w(env, n_loop = 300, max_n_step = 200, input_dim = env.observation_space.n)
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/model_taxi_dqn.h5")
    if plot_results:
        window_conv = 10
        reward_vec_conv = smoothen(reward_vec, window_conv)
        lc_conv = smoothen(lc, window_conv)
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(reward_vec_conv[window_conv:-window_conv], color = "black")
        axs[0].set_xlabel("trial number")
        axs[0].set_title("average reward")
        axs[1].set_title("average number of steps needed to finish")
        axs[1].plot(lc_conv[window_conv:-window_conv], color = "black")
        axs[1].set_xlabel("trial number")
    if train_model and solved:
        print("Problem solved.")
    if performance:
        perform(env, rl_agent, verbose = True)
    env.close()