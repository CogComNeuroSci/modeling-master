#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018

@author: tom verguts
Taxi!
This program teaches a taxi driver to pick up a client and drop him/her off
Algorithms are Rescorla-Wagner (rw); Sarsa; Sarsa-lambda; and Q-learning
All work fine except rw; this is because rw cannot "bridge" between current action
and later reward
 rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

algo = "ql"
env = gym.make('Taxi-v3')
n_episodes, max_per_episode = 500, 200
lr, gamma, lambd = 0.7, 0.95, 0.4
Q = np.random.rand(env.observation_space.n, env.action_space.n) # giant Q matrix for flat RL
tot_reward_epi, tot_finish = [], []
color_list = {"rw": "black", "sarsa": "red", "sarsalam": "blue", "ql": "green"}
window_conv = 10

# get to work
#env.render()
for ep in range(n_episodes):
    observation = env.reset()
    observation0 = env.observation_space.sample()
    action0 = env.action_space.sample()
    reward0 = np.random.randint(0, 1)
    tot_reward = 0
    if algo == "sarsalam":
        trace = np.zeros(env.observation_space.n)
    #print("episode {}".format(ep))
    for t in range(max_per_episode):
        t += 1
        #env.render() # show the maze
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
            #print("Episode finished after {} timesteps".format(t+1))
            break
    tot_reward /= t # average reward for this episode    
    #print("Task{}completed".format([" not ", " "][reward>0]))
    tot_reward_epi.append(tot_reward)
    tot_finish.append(t)

# show what you found; now inactive to speed up the "graphics" coming up next
#v_reward = np.convolve(tot_reward_epi,np.ones(window_conv)/window_conv)
#plt.subplot(121)
#plt.plot(v_reward[window_conv:-window_conv], color = color_list[algo])
#v_finish = np.convolve(tot_finish,np.ones(window_conv)/window_conv)
#plt.subplot(122)
#plt.plot(v_finish[window_conv:-window_conv], color = color_list[algo])

# and see it in action
observation = env.reset()
for t in range(10):
        t += 1
        env.render() # show the maze
        try:
            prob = np.exp(Q[observation,:])
            prob = prob/np.sum(prob)
            action = np.random.choice(range(env.action_space.n), p = prob) # softmax
        except:        
            action = env.action_space.sample() # random policy
        observation, reward, done, info = env.step(action)
        input() # for visual effect