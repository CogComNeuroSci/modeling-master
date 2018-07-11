#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018
Guessing game
beat the algorithm!
@author: tom
1 = too low
3 = too high
"""

import gym
import numpy as np
env = gym.make('GuessingGame-v0')
n_episodes = 100
max_try = 100
lower = -100
upper = +100
time_list = []
for i_episode in range(n_episodes):
    observation = env.reset()
    lowerb = lower
    upperb = upper
    for t in range(max_try):
        x = np.asarray([np.mean([lowerb,upperb])], dtype = float)
        #action = env.action_space.sample() # random sample; not so good...
        observation, reward, done, info = env.step(x)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        if observation == 3:
            upperb = np.mean([lowerb, upperb])
        if observation == 1:
            lowerb = np.mean([lowerb, upperb])
    if t == max_try-1:
       if observation == 3:
           lower = lower - 1000
       if observation == 1:
           upper = upper + 1000
    time_list.append(t+1)
print("mean number of {:.1f} guesses!".format(np.mean(time_list)))