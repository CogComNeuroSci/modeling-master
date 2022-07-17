#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
the mountain car problem with DQN (as in mnhih et al; uses double-DQN class AgentD)
works but not amazingly efficient
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from ch8_tf2_pole_1 import perform
from ch8_tf2_pole_2 import AgentD

def learn_w(n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4):
    lc = np.zeros(n_loop)
    buffer_count = 0
    threshold = 0.2
    stop_crit, success_crit = False, 10
    loop, success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
        data = np.zeros((max_n_step, input_dim*2 + 3)) # data for this loop
        while not done:
            action = rl_agent.sample_soft(state)
            next_state, reward, done, info = env.step(action)
            data[n_step, 0:input_dim] = state
            data[n_step, input_dim:2*input_dim] = next_state
            data[n_step, -3]  = action
#            fb = reward + int(done)*(n_step<(200-1))*50
            if state[0] > threshold: done = 1
            fb = reward + int(state[0]>threshold)*20
            data[n_step, -2]  = fb
            data[n_step, -1]  = done
            n_step += 1
            state = next_state
        buffer_count = rl_agent.update_buffer(data, n_step, buffer_count)
        if not loop % rl_agent.update_gran:
            rl_agent.update_q()
        if (not loop % rl_agent.learn_gran) and (buffer_count > 100): # don't learn first 500 trials
            rl_agent.learn(buffer_count, verbose = False)
        lc[loop] = n_step
        loop += 1
        success += (fb > 0)
        print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (success > success_crit)
    return lc, success > 10

        
if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    load_model, save_model, save_data, train_model = False, True, True, True
    rl_agent = AgentD(env.observation_space.shape[0], env.action_space.n, \
                           buffer_size = 500, epsilon_min = 0.01, epsilon_max = 1, \
                           epsilon_dec = 0.99, lr = 0.001, gamma = 0.99, learn_gran = 1, update_gran = 10, nhid1 = 64, nhid2 = 8)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(join(os.getcwd(), "models", "model_mountaincar.h5"))
    if train_model:
        lc, solved = learn_w(n_loop = 200, max_n_step = 200, input_dim = env.observation_space.shape[0])
    if save_model:
        tf.keras.models.save_model(rl_agent.network, join(os.getcwd(), "models", "model_mountaincar.h5"))
    if train_model:        
        plt.plot(lc)
    if save_data:
        
    if train_model and solved:
        print("Problem solved.")
    perform(env, rl_agent, verbose = False)
    env.close()