#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:18:00 2020

@author: tom verguts
as you can see... under construction
"""

import gym, time
env = gym.make("MsPacman-v0")

env.render()
state = env.reset()
done = False
max_n_step = 300
n_step = 0

while not done and n_step < max_n_step:
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    n_step += 1 
    time.sleep(0.1)

env.close()