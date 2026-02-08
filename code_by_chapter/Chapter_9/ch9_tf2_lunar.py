#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
solves the lunar problem from gymnasium
with DOUBLE deep q learning (DQN) and episode replay as described in Mnih et al (2015)
every learn_gran trials, a sample from the buffer is used to learn (deep) Q-values
a good setting is 2 (trained) hidden layers of 64 units, learning rate = .00001
"""

import gymnasium as gym
import tensorflow as tf
import numpy as np
import os
from ch8_tf2_pole_2 import AgentD
import matplotlib.pyplot as plt

def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4, success_crit: int = 10, verbose: bool = False):
    """learn the weights of the network"""
    lc = np.zeros(n_loop) # number of steps in the episode
    tw = np.zeros(n_loop) # reward obtained on the episode
    buffer_count = 0
    stop_crit = False
    loop, tot_success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step= 0
        state, _ = env.reset()
        data = np.zeros((max_n_step, input_dim*2 + 3)) # data for this loop
        termin, trunc = 0, 0
        tot_reward = 0
        while not (termin or trunc or n_step == max_n_step):
            action = rl_agent.sample_soft(state)
            next_state, reward, termin, trunc, info = env.step(action)
            data[n_step, 0:input_dim] = state
            data[n_step, input_dim:2*input_dim] = next_state
            data[n_step, -3]  = action
            data[n_step, -2]  = reward
            data[n_step, -1]  = int(trunc or termin)
            n_step += 1
            state = next_state
            tot_reward += reward
        buffer_count = rl_agent.update_buffer(data, n_step, buffer_count) # transfer current data to the buffer for later training
        if not loop % rl_agent.update_gran:
            rl_agent.update_q()
        if (not loop % rl_agent.learn_gran) and (buffer_count > 500):     # learn every learn_gran trials, but not the first 500 trials
            rl_agent.learn(buffer_count, verbose = True)
        lc[loop]  = n_step
        tw[loop] = tot_reward 
        loop += 1
        success = int(tot_reward >= 200)
        tot_success += success
        if verbose:
            print("success = ", success)
            print("tot success = ", tot_success)
            print("reward = ", tot_reward)
            print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (tot_success > success_crit)
    return lc, tw, (tot_success > success_crit)

def perform(env, rl_agent, verbose: bool = False):
    """after training, show how the model performs on the task"""
    env = gym.make('LunarLander-v2', render_mode = "human")
    state, _ = env.reset()
    n_step, termin, trunc = 0, 0, 0
    while not (termin or trunc): #  or bump_wall
        action = rl_agent.sample_soft(state)
        next_state, reward, termin, trunc, info = env.step(action)
        n_step += 1
        state = next_state
        if verbose:
            print(n_step)
    env.close()
	    
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    nhid1  = 64 
    nhid2  = 64
    train1 = True # should the hidden layer be fixed or trained
    sim_nr = 0    # for parameter sweeps 
    load_model, save_model, train_model = False, True, True
    rl_agent = AgentD(env.observation_space.shape[0], env.action_space.n, \
                           buffer_size = 1000, epsilon_min = 0.001, epsilon_max = 0.99, \
                           epsilon_dec = 0.999, lr = 0.00001, gamma = 0.995, learn_gran = 2, \
						   nhid1 = nhid1, nhid2 = nhid2, train1 = train1, train2 = train1, \
						   start2learn = 400, update_gran = 20)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/models/model_cartpole.keras")
    if train_model:
        lc, tw, solved = learn_w(env, n_loop = 4000, max_n_step = 2000, input_dim = env.observation_space.shape[0], success_crit = 20, verbose = True)
    if save_model:
        path = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(path):
            os.mkdir(path)	
        np.savez(path + "/res" + str(sim_nr), lc = lc, tw = tw)
    if train_model:
        if solved: print("Problem solved")
        fig, ax = plt.subplots(nrows = 1, ncols = 2)		
        ax[0].plot(lc)
        ax[1].plot(tw)
    perform(env, rl_agent, verbose = False)
    env.close()