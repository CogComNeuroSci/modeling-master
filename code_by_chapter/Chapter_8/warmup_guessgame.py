#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018
Guessing game: warming up environment for open AI gym
try to beat the bound algorithm!
@author: tom verguts
1 = too low
3 = too high
"""

import gym
import numpy as np

def update_bounds(observation, lowerb, upperb):
    if observation == 3:
        upperb = np.mean([lowerb, upperb])
    if observation == 1:
        lowerb = np.mean([lowerb, upperb])
    return lowerb, upperb

def update_outer_bounds(observation, lower, upper):
    # if you didn't find the number in max_try, cast your net more widely
    if observation == 3:
           lower = lower - 1000
    if observation == 1:
           upper = upper + 1000
    return lower, upper

env = gym.make('GuessingGame-v0')
n_episodes, max_try = 100, 100
lower, upper = -100, +100
time_list = []
algo_list = ["random", "bound"]
algo = "bound"

for i_episode in range(n_episodes):
    observation = env.reset()
    if algo == "bound":
        lowerb, upperb = lower, upper
    for t in range(max_try):
        if algo == "bound":
            x = np.asarray([np.mean([lowerb,upperb])], dtype = float)
        else:
            x = env.action_space.sample() # random sample
        observation, reward, done, info = env.step(x)
        if algo == "bound":
            lowerb, upperb = update_bounds(observation, lowerb, upperb)
        if done: break

    print("Episode finished after {} timesteps".format(t+1))
    if algo == "bound" and t == max_try-1:
        lower, upper = update_outer_bounds(observation, lower, upper)
    time_list.append(t+1)

print("mean number of {:.1f} guesses!".format(np.mean(time_list)))