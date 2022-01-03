#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate a value function using dynamic programming
this is for Q-values, in particular, equation on p 139 in MCP book (the equivalent eq is not shown in S & B 2018)
some states are slippery with probability slip
this is for lineworld
note: discount factor is called gamma in S & B, but eta in MCP book
"""
# %% import and initialization
import numpy as np
import matplotlib.pyplot as plt
from ch9_plotting import plot_value

np.set_printoptions(precision=4, suppress = True)

def succ(state_pass = 1, action_pass = 1): 
    """successor function"""
    return state + action_pass*2 - 1
    
nstates, nactions = 7, 2
r = [0.7, 0, 0, 0, 0, 0, 0.8]
slip = 0.1
Q_value = np.random.random((nstates, 2))
gamma = 0.2 # discount factor
stop, converge, threshold, max_iteration = False, False, 0.01, 10000
halfway = 5 # intermediate-step value matrix to be printed
fig, axs = plt.subplots(1, 1)
    
#%% main code
#plot_value(0, 0, value)
print(Q_value)
iteration = 0
action_prob = 1/2 # random policy

# start to iterate                
while stop == False:
    previous_Q_value = np.copy(Q_value)
    iteration += 1
    for state in range(nstates):
        for action in range(nactions):
			# this implements the bellman equation on p 139 in MCP book; note that eta is forgotten in that eq.
			# the if-elif implements the sum over s' (successor function); the sum on each line below implements the sum over a' 
            if (state==0): 
                action_Q = r[1] +         gamma*action_prob*(previous_Q_value[1, action] + previous_Q_value[1, 1-action]) 
            elif (state==nstates-1):
                action_Q = r[nstates-2] + gamma*action_prob*(previous_Q_value[nstates-2, action] + previous_Q_value[nstates-2, 1-action])
            elif state in (4, 5):
                action_Q = slip*(r[succ(state, 1-action)] + \
                                 gamma*action_prob*(previous_Q_value[succ(state,1-action),0]+previous_Q_value[succ(state,1-action),1])) + \
                           (1-slip)*(r[succ(state, action)]+ \
                                     gamma*action_prob*(previous_Q_value[succ(state,action),0]+previous_Q_value[succ(state,action),1]))
            else:    
                action_Q = r[succ(state,action)] + gamma*action_prob*(previous_Q_value[succ(state,action),0]+previous_Q_value[succ(state,action),1])
            Q_value[state, action] = action_Q
    if np.mean(np.abs(Q_value-previous_Q_value))<threshold:
        converge = stop = True
    elif iteration>max_iteration:
        stop = True

    
#%% show what you did
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
print(Q_value)
plot_value(fig, axs, 0, 0, Q_value, n = 0, grid = False, title = "Q values")
