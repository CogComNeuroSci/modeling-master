#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018

@author: tom verguts

This program uses the tabular agent class TabAgent from ch9_RL_taxi

Algorithms are all from the MDP approach (chapter 9 MCP book):
Rescorla-Wagner (rw); Sarsa (sarsa); Sarsa-lambda (sarsalam); and Q-learning (ql)

"""

#%% import and initialization
import gym
import numpy as np
import matplotlib.pyplot as plt
from ch9_RL_taxi import smoothen, TabAgent, update


def space2state(observation, xmin, xmax, ymin, ymax, gran):
    row = np.floor((observation[0]-xmin)/(xmax-xmin)*gran)
    col = np.floor((observation[1]-ymin)/(ymax-ymin)*gran)
    return (row*gran + col).astype(int)


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
        observation, reward, done, info = env.step(action)


#%% main code
if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    algo = "ql" # options are rw, sarsa, sarsalam, or ql
    n_episodes, max_per_episode = 5000, 200
    tot_reward_epi, tot_finish = [], []
    verbose = True # do you want to see intermediate results in optimisation
    granul = 10     # nr of levels per input dimension
    n_states = granul**env.observation_space.shape[0]
    rl_agent = TabAgent(n_states = n_states, n_actions = env.action_space.n,
                        algo = algo, lr = 0.05, gamma = 0.995, lambd = 0.4 ) # giant Q matrix for flat RL)
    xmin, xmax, ymin, ymax = np.min(env.observation_space.low[0]),  \
                              np.max(env.observation_space.high[0]),\
                              np.min(env.observation_space.low[1]), \
                              np.max(env.observation_space.high[1]) \

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
            observation1, reward, done, info = env.step(action)
            state0 = space2state(observation0, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                               gran = granul)
            state  = space2state(observation, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                              gran = granul)
            state1 = space2state(observation1, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                              gran = granul)
            if done and t < max_per_episode-1: reward = 10
            rl_agent.learn(state0, state, state1, action0, action, reward0, reward)
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
    color_list = {"rw": "black", "sarsa": "red", "sarsalam": "blue", "ql": "green"}
    window_conv = 10 # convolution window for smooth curves
    fig, axs = plt.subplots(1, 2)    
    v_reward = smoothen(tot_reward_epi, window_conv)
    axs[0].set_title("average reward obtained")
    axs[0].plot(v_reward[window_conv:-window_conv], color = color_list[algo])
    axs[0].set_xlabel("trial number")
    v_finish = smoothen(tot_finish, window_conv)
    axs[1].set_title("average number of steps needed to finish")
    axs[1].plot(v_finish[window_conv:-window_conv], color = color_list[algo])
    axs[1].set_xlabel("trial number")
    
    performance(env, rl_agent)
    env.close()
