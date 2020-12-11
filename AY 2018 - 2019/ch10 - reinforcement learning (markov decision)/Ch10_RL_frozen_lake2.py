#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018
simple RL model in python
action mapping: left, down, right, up

uses open AI gym, Q-learning (code partially stolen from the internet)
Sarsa, and rescorla-wagner algorithms for frozen lake walking
registers a non-slippery version of FrozenLake; is much easier to understand
registering should be done just once
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
#env = gym.make("FrozenLake-v0")
env = gym.make("FrozenLakeNotSlippery-v0")
np.set_printoptions(precision=3, suppress=True)
Q = np.zeros([env.observation_space.n,env.action_space.n])
algo = "sarsa" # qlearning, sarsa, or rescorla-wagner
lr = .8
gamma = .95
num_episodes= 1000
r_list = []
j_list = []
window = 100
for i in range(num_episodes):
    s = env.reset()
    s0 = env.observation_space.sample()
    a0 = env.action_space.sample()
    r0 = np.random.randint(0,1)
    rAll = 0
    d = False
    j = 0
    while j < 99:
#        env.render()
        j += 1
        a = np.argmax(Q[s,:]+np.random.randn(1,env.action_space.n)*(1./(i+1)))
        s1, r, d, _ = env.step(a)
        if algo == "qlearning":
            backup = r + gamma*np.max(Q[s1,:])
            Q[s,a] = Q[s,a] + lr*(backup - Q[s,a])
        elif algo == "sarsa":
            backup = r0 + gamma*Q[s,a]
            Q[s0,a0] = Q[s0,a0] + lr*(backup - Q[s0,a0])
        else: # rescorla-wagner in this case
            backup = r 
            Q[s,a] = Q[s,a] + lr*(backup - Q[s,a])
        rAll += r
        s0 = s
        s = s1
        a0 = a
        r0 = r
        if d == True:
            break
    if algo == "sarsa":    # one last update needed for sarsa to "feel" reward
        backup = r0 + gamma*Q[s,a]
        Q[s0,a0] = Q[s0,a0] + lr*(backup - Q[s0,a0])
    r_list.append(rAll)     
    j_list.append(j)
print("Average episode length:" + str(sum(j_list[-window:-1])/window))
print("Score over time: " +  str(sum(r_list[-window:-1])/window))
print(Q)