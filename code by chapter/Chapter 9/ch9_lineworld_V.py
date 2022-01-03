#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate a value function using dynamic programming
in particular, equation (3.14) from S&B
some states are slippery with probability slip
this is for V-values in lineworld
in lineworld (not in gridworld), V and Q definitions are consistent with S&B notation. In particular, V and Q 
values are calculated for SUBSEQUENT states only; current state is never included
"""
# %% import and initialize

import numpy as np
import matplotlib.pyplot as plt
from ch9_plotting import plot_value

np.set_printoptions(precision=4, suppress = True)

def succ(state_pass = 1, action_pass = 1): # successor function (p(s',r / s,a))
    return state + action_pass*2 - 1
    
nstates = 7
r = [0.3, 0, 0, 0, 0, 0, 0.8] # reward in each state
slip = 0.8
value = np.random.random(nstates)
gamma = 0.4 # discount factor
stop, converge, threshold, max_iteration = False, False, 0.01, 10000
halfway = 5 # intermediate-step value matrix to be printed
fig, axs = plt.subplots(1, 1)
    
#%% start to iterate
action_prob = 1/2 # random policy
print(value)
iteration = 0
while stop == False:
    previous_value = value + 0
    iteration += 1
    for state in range(nstates):
        total_v = 0
        for action in range(2):
            if (state==0):
                action_v = r[1] + gamma*previous_value[1]
            elif (state==nstates-1):
                action_v = r[nstates-2] + gamma*previous_value[nstates-2]
            elif state in (4, 5): # the slippery states
                action_v = slip*    (r[succ(state,1-action)] + gamma*previous_value[succ(state,1-action)]) + \
                          (1-slip)* (r[succ(state,action)]   + gamma*previous_value[succ(state,action)])
            else:    
                action_v = r[succ(state,action)] + gamma*previous_value[succ(state,action)]
            action_v *= action_prob
            total_v += action_v
        value[state] = total_v
    if np.mean(np.abs(value-previous_value))<threshold:
        converge = stop = True
    elif iteration>max_iteration:
        stop = True
    if iteration == halfway:
        #plot_value(1, 0, value)
        pass
    
#%% plot results
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
plot_value(fig, axs, 0, 0, value, title = "V", n = 0)
print(value)