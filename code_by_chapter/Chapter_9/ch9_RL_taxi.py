#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018

@author: tom verguts
Taxi!
This program teaches a taxi driver to pick up a client and drop him/her off
see the AI gym website for more info
Algorithms are all from the MDP approach (chapter 9 MCP book):
Rescorla-Wagner (rw); Sarsa (sarsa); Sarsa-lambda (sarsalam); and Q-learning (ql)
All work fine except rw; this is because rw cannot "bridge" between current action
and later reward. See MCP book for detailed explanation why.
Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations
"""

#%% import and initialization
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v2')
algo = "sarsa" # options are rw, sarsa, sarsalam, or ql
n_episodes, max_per_episode = 200, 200
lr, gamma, lambd = 0.7, 0.95, 0.4 # learning rate, discount rate, eligibility trace (lambda)
Q = np.random.rand(env.observation_space.n, env.action_space.n) # giant Q matrix for flat RL
tot_reward_epi, tot_finish = [], []
color_list = {"rw": "black", "sarsa": "red", "sarsalam": "blue", "ql": "green"}
window_conv = 10 # convolution window for smooth curves
verbose = True # do you want to see intermediate results in optimisation

def smoothen(vector, window):
    return np.convolve(vector, np.ones(window)/window)

#%% main code
for ep in range(n_episodes):
    observation = env.reset()
    observation0 = env.observation_space.sample()
    action0 = env.action_space.sample()
    reward0 = np.random.randint(0, 1)
    tot_reward = 0
    if algo == "sarsalam":
        trace = np.zeros(env.observation_space.n)
    if verbose:
        print("episode {}".format(ep))
    for t in range(max_per_episode):
        try:
            prob = np.exp(Q[observation,:])
            prob = prob/np.sum(prob)
            action = np.random.choice(range(env.action_space.n), p = prob) # softmax
        except:        
            action = env.action_space.sample() # random policy
        observation1, reward, done, info = env.step(action)
        if algo == "rw":
            backup = reward
            Q[observation, action] += lr*(backup - Q[observation, action])
        elif algo == "sarsa":
            backup = reward0 + gamma*Q[observation, action]
            Q[observation0, action0] += lr*(backup - Q[observation0, action0])
        elif algo == "sarsalam":
            backup = reward0 + gamma*Q[observation, action]
            Q[:, action0] += lr*(backup - Q[observation0, action0])*trace
        else: # q-learning
            backup = reward + gamma*np.max(Q[observation1, :])
            Q[observation, action] += lr*(backup - Q[observation, action])
        if algo == "sarsalam": # decaying trace
            v = np.zeros(env.observation_space.n)
            v[observation] = 1
            trace = gamma*lambd*trace + v
        observation0 = observation # previous state
        observation = observation1
        action0 = action           # previous action
        reward0 = reward
        tot_reward += reward
        if done:
            if verbose:
                print("Episode finished after {} timesteps".format(t+1))
            break
    tot_reward /= t # average reward for this episode    
    if verbose:
        print("Task{}completed".format([" not ", " "][reward>0]))
    tot_reward_epi.append(tot_reward)
    tot_finish.append(t)

#%% plot results
fig, axs = plt.subplots(1, 2)

v_reward = smoothen(tot_reward_epi, window_conv)
axs[0].set_title("average reward obtained")
axs[0].plot(v_reward[window_conv:-window_conv], color = color_list[algo])
axs[0].set_xlabel("trial number")

v_finish = smoothen(tot_finish, window_conv)
axs[1].set_title("average number of steps needed to finish")
axs[1].plot(v_finish[window_conv:-window_conv], color = color_list[algo])
axs[1].set_xlabel("trial number")

# do you want to see the process live, and if so how many steps
# press Enter in the console to proceed to the next state
see_live, n_steps = False, 5 
if see_live:
    observation = env.reset()
    for t in range(n_steps):
        t += 1
        env.render() # show the maze
        try:
            prob = np.exp(Q[observation,:])
            prob = prob/np.sum(prob)
            action = np.random.choice(range(env.action_space.n), p = prob) # softmax
        except:        
            action = env.action_space.sample() # random policy
        observation, reward, done, info = env.step(action)
        input() # press Enter in the console to proceed to the next state