#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:28:40 2018

@author: tom verguts
n-armed bandit (n = len(p))
solved with 
- gradient ascent (chap 8)
- epsilon-greedy algorithm (chap 9)
- optimistic starts (chap 9)
"""
#%% import and initialization
import numpy as np
import matplotlib.pyplot as plt

# available algorithms: gradient ascent, epsilon-greedy, optimistic starts
algo = "gradient" # options: gradient, epsilon, optimist
np.set_printoptions(precision=2)
n_trials = 1000
beta = 2 # learning rate
p = [0.2, 0.4, 0.6, 0.8] # payoff probabilities
w = np.random.random(len(p)) # can be interpreted as weights or Q-values
if algo == "optimist": # optimistic starts (implemented via initial weights w)
    w += 5
    epsilon = 0.0
elif algo=="epsilon": # make a completely random choice with probability epsilon
    epsilon = 0.1
r= []            # reward list
window = 30      # window for reward calculation
window_conv = 20 # window size for convolution; plots will be smoother with larger window_conv
threshold = 0.8  # check whether (convolved) reward is above this threshold (the earlier, the more efficieent the algorithm)
color_list = {"gradient": "black", "epsilon": "red", "optimist": "blue"}

#%% let's play: different algorithms
for loop in range(n_trials):
    if algo == "gradient":
        prob = np.exp(beta*w)
        prob = prob/np.sum(prob)
        choice = np.random.choice(range(len(p)),p = prob)
    else: # epsilon-greedy or optimistic starts
        if np.random.random()>epsilon:
            choice = np.argmax(w)
        else:
            choice = np.random.choice(range(len(w)))
    r.append(np.random.choice([0, 1], p =  [1 - p[choice], p[choice]])) # reward with prob p[choice]
    if algo == "gradient":
		# apply gradient asccent algorithm (REINFORCE) (chap 8)
        w += np.asarray((range(len(p))==choice)-prob)*beta*r[-1]
    else:
		# calculate average reward for option choice; (chap 9)
		# strictly speaking, update rate 1/(loop+1) should be tracked for each choice separately
		# alternative, one could put update rate a constant (and thus take an exponentially weighted average)
        w[choice] += 1/(loop+1)*(r[-1]-w[choice]) 

#%% print and plot results
print(["weights",w])
print("mean reward: {:.2f}".format(np.mean(r[-window:])))
v = np.convolve(r,np.ones(window_conv)/window_conv)
hit_point = np.min([i for i in range(len(v)) if v[i]>threshold])
print("first point above threshold: {:.2f}".format(hit_point))
plt.plot(v[:-window_conv], color = color_list[algo])