#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018

@author: tom
Taxi!
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

algo = "rw"
env = gym.make('Taxi-v2')
n_episodes, max_per_episode = 1000, 200
lr, gamma = 0.7, 0.95
n_action = 6
Q = np.random.rand(5*5*5*4, n_action) # giant Q matrix for flat RL
tot_reward_epi, tot_finish = [], []

# get to work
for ep in range(n_episodes):
    observation = env.reset()
    observation0 = env.observation_space.sample()
    action0 = env.action_space.sample()
    reward0 = np.random.randint(0, 1)
    done = False
    tot_reward = 0
    print("episode {}".format(ep))
    for t in range(max_per_episode):
        t += 1
#       env.render() # show the maze
        try:
            prob = np.exp(Q[observation,:])
            prob = prob/np.sum(prob)
        except:
            prob = range(n_action)/n_action
        action = np.random.choice(range(n_action), p = prob) # softmax
        #action = env.action_space.sample() # random policy
        observation1, reward, done, info = env.step(action)
        #print(reward)
        if algo == "rw":
            backup = reward
            Q[observation, action] += lr*(backup - Q[observation0, action])
        elif algo == "sarsa":
            backup = reward0 + gamma*Q[observation,action]
            Q[observation0, action0] += lr*(backup - Q[observation0, action0])
        else: # q-learning
            backup = reward + gamma*np.max(Q[observation1, :])
            Q[observation, action] += lr*(backup - Q[observation, action])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        observation0 = observation # previous state
        observation = observation1
        action0 = action           # previous action
        reward0 = reward
        tot_reward += reward
    tot_reward /= t # average reward for this episode    
    print("Task{}completed".format([" not ", " "][reward>0]))
    tot_reward_epi.append(tot_reward)
    tot_finish.append(t)

# show what you found
plt.subplot(121)
plt.plot(range(n_episodes),tot_reward_epi)
plt.subplot(122)
plt.plot(range(n_episodes),tot_finish)