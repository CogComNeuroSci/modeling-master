#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:35:35 2022

@author: tom verguts
script for continuous action time mountain car
action space is discretized in order to apply tabular learning
"""

#%% import and initialization
import gym
import numpy as np
from ch9_RL_taxi import TabAgent, update, plot_results
from ch9_RL_mountaincar import space2state

def action2space(action, zmin, zmax, gran_action):
    epsilon = 0.001 # avoiding the action bounds
    return (zmax - zmin - 2*epsilon)*(action/(gran_action-1)) + zmin + epsilon

def performance(env, rl_agent: TabAgent):
    """"
    do you want to see the process live
    """
    observation = env.reset()
    done = False
    while not done:
        env.render() # show the maze
        state = space2state(observation, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                             gran = granul)
        action = rl_agent.safe_softmax(env, state)
        cont_action = action2space(action, zmin, zmax, granul_action)
        observation, reward, done, info = env.step([cont_action])
        
#%% main code
if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    algo = "ql" # options are rw, sarsa, sarsalam, or ql
    n_episodes, max_per_episode = 2000, 1000
    tot_reward_epi, tot_finish = [], []
    verbose = True # do you want to see intermediate results in optimisation
    granul = 10         # nr of levels per input dimension
    granul_action = 4   # nr of action levels
    n_states = granul**env.observation_space.shape[0]
    rl_agent = TabAgent(n_states = n_states, n_actions = granul_action,
                        algo = algo, lr = 0.05, gamma = 0.995, lambd = 0.4 ) # giant Q matrix for flat RL)
    xmin, xmax, ymin, ymax = np.min(env.observation_space.low[0]),  \
                              np.max(env.observation_space.high[0]),\
                              np.min(env.observation_space.low[1]), \
                              np.max(env.observation_space.high[1])
    zmin, zmax = np.min(env.action_space.low), np.max(env.action_space.high)
    for ep in range(n_episodes):
        if verbose:
            print("episode {}".format(ep))
        observation = env.reset()
        observation0 = env.observation_space.sample()
        action0 = env.action_space.sample()
        reward0 = np.random.randint(0, 1)
        tot_reward = 0
        for t in range(max_per_episode):
            state = space2state(observation, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                             gran = granul)
            action = rl_agent.safe_softmax(env, state)
            cont_action = action2space(action, zmin, zmax, granul_action)
            observation1, reward, done, info = env.step([cont_action])
            state0 = space2state(observation0, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                               gran = granul)
            state  = space2state(observation, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                              gran = granul)
            state1 = space2state(observation1, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                              gran = granul)
            cont_action0 = action2space(action0, zmin, zmax, granul_action)
#            if done and t < max_per_episode-1: reward = 10
            rl_agent.learn(state0, state, state1, action0, action, reward0, reward)
            rl_agent.Q = np.minimum(np.maximum(-100, rl_agent.Q), 100)
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
    plot_results(tot_reward_epi, tot_finish, algo)
    performance(env, rl_agent)
    env.close()