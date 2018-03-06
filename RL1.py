#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018
simple RL model in python
uses open AI gym and Q-learning (code partially stolen from the internet)
registers a non-slippery version of FrozenLake; is much easier to understand
registering should be done just once
question for students: Why doesn't rescorla wagner work in this environment?
@author: tom
how to register the non-slippery version:
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
"""

import gym
import numpy as np
env = gym.make("FrozenLakeNotSlippery-v0")
np.set_printoptions(precision=3, suppress=True)
Q = np.zeros([env.observation_space.n,env.action_space.n])
algo = "qlearning" # if not qlearning, then Rescorla-Wagner is used
lr = .8
gamma = .95
num_episodes= 200
r_list = []
j_list = []
window = 100
for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        a = np.argmax(Q[s,:]+np.random.randn(1,env.action_space.n)*(1./(i+1)))   
        s1, r, d, _ = env.step(a)
        if algo == "qlearning":
            backup = r + gamma*np.max(Q[s1,:])
        else:
            backup = r 
        Q[s,a] = Q[s,a] + lr*(backup - Q[s,a]) 
        rAll += r
        s = s1
        if d == True:
            break
    r_list.append(rAll)     
    j_list.append(j)
print("Average episode length:" + str(sum(j_list[-window:-1])/window))
print("Score over time: " +  str(sum(r_list[-window:-1])/window))
print(Q)