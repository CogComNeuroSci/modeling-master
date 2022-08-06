#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:21:47 2022

@author: tom verguts
actor-critic for taxi
"""

import gym
from ch9_RL_taxi import TabAgent, update, performance, plot_results
import numpy as np

class TabACAgent(TabAgent):
    # tabular actor-critic agent
    def __init__(self, n_states, n_actions, algo, lr, gamma = 0, lambd = 0):
        super().__init__(n_states, n_actions, algo, lr, gamma, lambd)
        self.V = np.random.rand(n_states, 1) # the critic network
    
    def learn(self, observation0, observation, observation1, action0, action, reward0, reward, done):    
        super().learn(observation0, observation, observation1, action0, action, reward0, reward, done)
        if self.algo == "ac":
            backup = reward + self.gamma*self.V[observation1]*int(1-done)
            prediction_error = (backup - self.V[observation])
            self.V[observation] += self.lr*prediction_error
            self.Q[observation, action] += self.lr*prediction_error

#%% main code
if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    algo = "ac" # options are rw, sarsa, sarsalam, or ql
    n_episodes, max_per_episode = 500, 200
    tot_reward_epi, tot_finish = [], []
    verbose = True # do you want to see intermediate results in optimisation
    rl_agent = TabACAgent(n_states = env.observation_space.n, n_actions = env.action_space.n,
                        algo = algo, lr = 0.3, gamma = 0.95, lambd = 0.4 ) # giant Q matrix for flat RL
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
            rl_agent.learn(observation0, observation, observation1, 
                           action0, action, reward0, reward, done)
            observation0, observation, action0, reward0 = \
                                update(observation, observation1, action, reward)
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
    
    #%% show results
    plot_results(tot_reward_epi, tot_finish, algo, color_list = {"ac": "black"})
    see_live, n_steps = False, 5 
    if see_live:
        performance(env, rl_agent, n_steps)
    env.close()	