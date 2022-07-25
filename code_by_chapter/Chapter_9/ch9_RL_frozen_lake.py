#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018
simple RL model in python for manoeuvring across a frozen lake
action mapping: left, down, right, up

uses class TabAgent (tabular agent) orginally used for taxi problem
registers a non-slippery version of FrozenLake; is much easier to understand
registering should be done just once
@author: tom verguts
"""

import gym
import numpy as np
from ch9_RL_taxi import TabAgent, update, plot_results, performance

def register_non_slip():
    # register the non-slippery version (only do it once)
    from gym.envs.registration import register
    register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100),


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    slippery, need_to_register = False, False
    if slippery:
        env = gym.make("FrozenLake-v0")
    else:
        if need_to_register: register_non_slip()
        env = gym.make("FrozenLakeNotSlippery-v0")

    algo = "sarsa" # options are rw, sarsa, sarsalam, or ql
    n_episodes, max_per_episode = 1000, 100
    tot_reward_epi, tot_finish = [], []
    verbose = True # do you want to see intermediate results in optimisation
    rl_agent = TabAgent(n_states = env.observation_space.n, n_actions = env.action_space.n,
                        algo = algo, lr = 0.5, gamma = 0.9, lambd = 0.2) 
    for ep in range(n_episodes):
        if verbose:
            print("episode {}".format(ep))
        observation = env.reset()
        observation0 = env.observation_space.sample()
        action0 = env.action_space.sample()
        reward0 = np.random.randint(0, 1)
        tot_reward = 0
        for t in range(max_per_episode):
            action = rl_agent.safe_softmax(env, observation)
            observation1, reward, done, info = env.step(action)
            reward *= 10
            rl_agent.learn(observation0, observation, observation1, 
                           action0, action, reward0, reward, done)
            observation0, observation, action0, reward0 = \
                                update(observation, observation1, action, reward)
            tot_reward += reward
            if done:
                if verbose:
                    print("Episode finished after {} timesteps".format(t+1))
                break
        tot_finish.append(t)
        tot_reward_epi.append(tot_reward)

    # show results
    plot_results(tot_reward_epi, tot_finish, algo)
    see_live, n_steps = True, 5 
    if see_live:
        performance(env, rl_agent, n_steps, wait_input = False)
