#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:28:40 2018

@author: tom
n-armed bandit
solved with 
- gradient ascent (chap 9)
- epsilon-greedy algorithm (chap 10)
- optimistic starts (chap 10)
"""
# come in
import numpy as np
import matplotlib.pyplot as plt

# for starters
# available algorithms: gradient ascent, epsilon-greedy, optimistic starts
algo = "gradient" # options: gradient, epsilon, optimist
np.set_printoptions(precision=2)
ntrial = 1000
alpha = 0.5
beta = 2
p = [0.2, 0.4, 0.6, 0.8] # payoff probabilities
w = np.random.random(len(p)) # can be interpreted as weights or Q-values
if algo == "optimist":
    w += 5
r= []
window = 30
window_conv = 20
threshold = 0.8
if algo=="optimist":
    epsilon = 0
else:
    epsilon = 0.1
color_list = {"gradient": "black", "epsilon": "red", "optimist": "blue"}

# let's play
for loop in range(ntrial):
    if algo == "gradient":
        prob = np.exp(beta*w)
        prob = prob/np.sum(prob)
        choice = np.random.choice(range(len(p)),p=prob)
    else:
        if np.random.random()>epsilon:
            choice = np.argmax(w)
        else:
            choice = np.random.choice(range(len(w)))
    r.append(np.random.choice([0,1],p=[1-p[choice], p[choice]]))
    if algo == "gradient":
        w += np.asarray((range(len(p))==choice)-prob)*alpha*beta*r[-1]
    else:
        w[choice] += 1/(loop+1)*(r[-1]-w[choice])

# end game
print(["weights",w])
print("mean reward: {:.2f}".format(np.mean(r[-window:])))
v = np.convolve(r,np.ones(window_conv)/window_conv)
hit_point = np.min([i for i in range(len(v)) if v[i]>threshold])
print("first point above threshold: {:.2f}".format(hit_point))
plt.plot(v[:-window_conv], color = color_list[algo])