#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate the optimal value function using dynamic programming
(i.e., the value function of the optimal policy)
in particular, equation (3.19) from Sutton & Barto (2018)
some states are slippery with probability slip
this is for lineworld
note: discount factor is called gamma in S & B, but eta in MCP book
"""
# %% import and initialize
import numpy as np
import matplotlib.pyplot as plt
from ch9_plotting import plot_value

np.set_printoptions(precision=4, suppress = True)

def succ(state_pass = 1, action_pass = 1):
	"""The successor function"""
    return state + action_pass*2 - 1
    
nstates = 7
r = [0.7, 0, 0, 0, 0, 0, 0.8] # reward in each state
slip = 0.0
value = np.random.random(nstates)
gamma = 0.2 # discount factor
stop, converge, threshold, max_iteration = False, False, 0.01, 10000
halfway = 5 # intermediate-step value matrix to be printed
fig, axs = plt.subplots(1, 1)
    
#%% main code
# values before estimation
print(value)
iteration = 0

# start to iterate
while stop == False:
    previous_value = value + 0
    iteration += 1
    for state in range(nstates):
        total_v = []
        for action in range(2):
            if (state==0): # leftmost state
                action_v = r[1] + gamma*previous_value[1]
            elif (state==nstates-1): # rightmost states
                action_v = r[nstates-2] + gamma*previous_value[nstates-2]
            elif state in (4, 5): # slippery states
                action_v = slip*    (r[succ(state,1-action)] + gamma*previous_value[succ(state,1-action)]) + \
                          (1-slip)* (r[succ(state,action)]   + gamma*previous_value[succ(state,action)])
            else:    
                action_v = r[succ(state,action)] + gamma*previous_value[succ(state,action)]
            total_v.append(action_v)
        value[state] = max(total_v) # take the max here, rather than the average across actions; this is why it's the V of the optimal policy
    if np.mean(np.abs(value-previous_value))<threshold:
        converge = stop = True
    elif iteration>max_iteration:
        stop = True
#    if iteration == halfway:
#        plot_value(1, 0, value)
    
#%% show what you did
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
plot_value(fig, axs, 1, 1, value, n = 0)
print(value)