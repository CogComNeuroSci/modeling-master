#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
solves the cart pole problem with deep q learning and episode replay
this version addditionally uses the target network trick, also described by mnih et al
i was inspired by machine learning with phil for this implementation
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from ch8_tf2_pole_1 import Agent, perform, build_network


class AgentD(Agent):
    """
    a double-Q class; double as in, one network is used for the policy, the other network
    provides the targets (network_target). after every update_gran steps, the target network
    is set to the policy network
    """
    def __init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, 
                 epsilon_dec, lr, gamma, learn_gran, update_gran, nhid1, nhid2):
        Agent.__init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, 
                 epsilon_dec, lr, gamma, learn_gran, nhid1, nhid2)
        self.update_gran = update_gran
        self.network_target = build_network(self.n_states, self.n_actions, self.lr, self.nhid1, self.nhid2) # a second (double, target) network needed        

    def learn(self, n: int, verbose: bool = True): # this method is overwritten from Agent
        #self.epsilon = self.epsilon_max # in case you want to reset epsilon on each episode
        sample_size = np.minimum(100, n)
        sample = np.random.choice(n, sample_size)
        q = self.network.predict(self.x_buffer[sample])
        q_next = self.network.predict(self.xn_buffer[sample])
        q_next_target = self.network_target.predict(self.xn_buffer[sample])
        index_max = np.argmax(q_next, axis = 1) # what would the online network choose?
        q_max = q_next_target[list(range(q.shape[0])), index_max] # what does the target network think of this?
        include_v = 1 - self.d_buffer[sample]
        if verbose:
            print("x buffer:", self.x_buffer)
            print("n: ", n)
            print("q_predict", q)
        q_target = q.copy() 
        target_indices = np.dot(self.y_buffer[sample], np.arange(self.n_actions)).astype(int)
        q_target[list(range(q.shape[0])), target_indices] = np.squeeze(self.r_buffer[sample])
        q_target[list(range(q.shape[0])), target_indices] += self.gamma*q_max * np.squeeze(include_v)
       	self.network.fit(self.x_buffer[sample], q_target, batch_size = 64, epochs = 2000, verbose = 0)	
        if verbose:
            print("q_target", q_target)

    def update_q(self):
        self.network_target.set_weights(self.network.get_weights())


def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4, success_crit: int = 10):
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
        if not loop % rl_agent.update_gran:
            rl_agent.update_q()
        if (not loop % rl_agent.learn_gran) and (buffer_count > 0): # don't learn first 500 trials
            rl_agent.learn(buffer_count, verbose = False)
        lc[loop] = n_step
        loop += 1
        success += (n_step == max_n_step)
        print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (success > success_crit)
    return lc, success > success_crit


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    load_model, save_model, train_model = False, False, True
    rl_agent = AgentD(env.observation_space.shape[0], env.action_space.n, \
                           buffer_size = 1000, epsilon_min = 0.001, epsilon_max = 0.99, \
                           epsilon_dec = 0.999, lr = 0.001, gamma = 0.99, learn_gran = 1, update_gran = 5, nhid1 = 16, nhid2 = 8)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/models/model_cartpole.h5")
    if train_model:
        lc, solved = learn_w(env, n_loop = 50, max_n_step = 200, input_dim = env.observation_space.shape[0])
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/models/model_cartpole.h5")
    if train_model:
        plt.plot(lc)
    if train_model and solved:
        print("Problem solved.")
    perform(env, rl_agent, verbose = False)
    env.close()