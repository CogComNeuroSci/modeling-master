#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate a value function using dynamic programming
in particular, equation (3.14) from S&B
some states are slippery with probability slip
this is for Q-values
this is for lineworld
"""
import numpy as np
#import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress = True)

# %% preliminaries

def succ(state_pass = 1, action_pass = 1): # successor function
    return state + action_pass*2 - 1
    
#def plot_value(row, column, value_matrix):
#    offset = 0.02
#    for loop in range(nstates):
#        axs[row, column].text(offset + loop/7, 0.5, "{:.1f}".format(value_matrix[loop]))
#    axs[row, column].grid(True)
#    v = np.linspace(0,1, num = nstates+1)
#    axs[row, column].set_xticks(ticks = v)
#    axs[row, column].set_yticks(ticks = [0.45, 0.65])
#    axs[row, column].set_ylim(bottom = 0, top = 1)
#    axs[row, column].set_xticklabels(" ")
#    axs[row, column].set_yticklabels(" ")
#    return

nstates, nactions = 7, 2
r = [0.7, 0, 0, 0, 0, 0, 0.8]
slip = 0.45
Q_value = np.random.random((nstates, 2))
gamma = 0.8 # discount factor
stop, converge, threshold, max_iteration = False, False, 0.01, 10000
halfway = 5 # intermediate-step value matrix to be printed
#fig, axs = plt.subplots(2, 2)
    
#%% start to iterate
#plot_value(0, 0, value)
print(Q_value)
iteration = 0
action_prob = 1/2 # random policy
                
while stop == False:
    previous_Q_value = Q_value + 0
    iteration += 1
    for state in range(nstates):
        for action in range(nactions):
            if (state==0):
                action_Q = r[1] +         gamma*np.max((previous_Q_value[1, action], previous_Q_value[1, 1-action]))
            elif (state==nstates-1):
                action_Q = r[nstates-2] + gamma*np.max((previous_Q_value[nstates-2, action], previous_Q_value[nstates-2, 1-action]))
            elif state in (4, 5):
                action_Q = slip*(r[succ(state, 1-action)] + \
                                 gamma*np.max((previous_Q_value[succ(state,1-action),0], previous_Q_value[succ(state,1-action),1]))) + \
                           (1-slip)*(r[succ(state, action)]+ \
                                     gamma*np.max((previous_Q_value[succ(state,action),0], previous_Q_value[succ(state,action),1])))
            else:    
                action_Q = r[succ(state,action)] + gamma*np.max((previous_Q_value[succ(state,action),0], previous_Q_value[succ(state,action),1]))
            Q_value[state, action] = action_Q
    if np.mean(np.abs(Q_value-previous_Q_value))<threshold:
        converge = True
        stop = True
    elif iteration>max_iteration:
        stop = True
    else:
        pass
    if iteration == halfway:
        #plot_value(1, 0, value)
        pass
    
#%% show what you did
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
#plot_value(1, 1, value)
print(Q_value)