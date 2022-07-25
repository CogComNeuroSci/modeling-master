#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:03:05 2018

@author: tom verguts
Taxi!
This program teaches a taxi driver to pick up a client and drop him/her off
see the AI gym website for more info
This program introduces class TabAgent

Algorithms are all from the MDP approach (chapter 9 MCP book):
Rescorla-Wagner (rw); Sarsa (sarsa); Sarsa-lambda (sarsalam); and Q-learning (ql)
All work fine except rw; this is because rw cannot "bridge" between current action
and later reward. See MCP book for detailed explanation why.
Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations
"""

#%% import and initialization
import gym
import numpy as np
import matplotlib.pyplot as plt

def smoothen(vector, window):
    return np.convolve(vector, np.ones(window)/window)

class TabAgent(object):
    """
    a tabular RL agent; 
    can do Rescorla-Wagner (rw), SARSA (sarsa), SARSA-lambda (sarsalam), and Q-learning (ql)
    """
    def __init__(self, n_states, n_actions, algo, lr, gamma = 0, lambd = 0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions = np.arange(n_actions)
        self.algo = algo
        self.Q = np.random.rand(n_states, n_actions) # giant Q matrix for flat RL
        self.lr = lr
        self.gamma = gamma # discount rate 
        self.lambd = lambd # eligibility trace parameter (for TD-lambda)
        if self.algo == "sarsalam":
            self.trace = np.zeros(self.n_states)
            
    def learn(self, observation0, observation, observation1, action0, action, reward0, reward, done):    
        if self.algo == "rw":
            backup = reward
            self.Q[observation, action] += self.lr*(backup - self.Q[observation, action])
        elif self.algo == "sarsa":
            backup = reward0 + self.gamma*self.Q[observation, action]*int(1-done)
            self.Q[observation0, action0] += self.lr*(backup - self.Q[observation0, action0])
        elif self.algo == "sarsalam":
            backup = reward0 + self.gamma*self.Q[observation, action]*int(1-done)
            self.Q[:, action0] += self.lr*(backup - self.Q[observation0, action0])*self.trace
        else: # q-learning
            backup = reward + self.gamma*np.max(self.Q[observation1, :])*int(1-done)
            self.Q[observation, action] += self.lr*(backup - self.Q[observation, action])
        if self.algo == "sarsalam": # decaying trace
            v = np.zeros(self.n_states)
            v[observation] = 1
            self.trace = self.gamma*self.lambd*self.trace + v

    def safe_softmax(self, env, observation):
        try:
            prob = np.exp(self.Q[observation,:])
            prob = prob/np.sum(prob)
            action = np.random.choice(range(self.n_actions), p = prob) # softmax
        except:        
            action = env.action_space.sample() # random policy
        return action    

def update(observation, observation1, action, reward):
    return observation , observation1, action, reward

def performance(env, rl_agent: TabAgent, n_steps: int = 100, wait_input: bool = True):
    """"
    do you want to see the process live, and if so how many steps
    wait_input: press Enter in the console to proceed to the next state
    """
    observation = env.reset()
    for t in range(n_steps):
        t += 1
        env.render() # show the maze
        action = rl_agent.safe_softmax(env, observation)
        observation, reward, done, info = env.step(action)
        if wait_input:
            input() # press Enter in the console to proceed to the next state

def plot_results(tot_reward_epi, tot_finish, algo):
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

#%% main code
if __name__ == "__main__":
    env = gym.make('Taxi-v2')
    algo = "ql" # options are rw, sarsa, sarsalam, or ql
    n_episodes, max_per_episode = 200, 200
    tot_reward_epi, tot_finish = [], []
    verbose = True # do you want to see intermediate results in optimisation
    rl_agent = TabAgent(n_states = env.observation_space.n, n_actions = env.action_space.n,
                        algo = algo, lr = 0.7, gamma = 0.95, lambd = 0.4 ) # giant Q matrix for flat RL)
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
    plot_results(tot_reward_epi, tot_finish, algo)
    see_live, n_steps = False, 5 
    if see_live:
        performance(env, rl_agent, n_steps)
