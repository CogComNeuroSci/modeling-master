#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate a value function using dynamic programming
in particular, equation (3.14) from S&B
some states are slippery with probability slip
this is for lineworld
"""
import numpy as np
import matplotlib.pyplot as plt
from ch10_plotting import plot_value

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

nstates = 7
r = [0.7, 0, 0, 0, 0, 0, 0.8]
slip = 0.0
value = np.random.random(nstates)
gamma = 0.2 # discount factor
stop, converge, threshold, max_iteration = False, False, 0.01, 10000
halfway = 5 # intermediate-step value matrix to be printed
#fig, axs = plt.subplots(2, 2)
    
#%% start to iterate
#plot_value(0, 0, value)
print(value)
iteration = 0
action_prob = 1/2 # random policy
while stop == False:
    previous_value = value + 0
    iteration += 1
    for state in range(nstates):
        total_v = []
        for action in range(2):
            if (state==0):
                action_v = r[1] + gamma*previous_value[1]
            elif (state==nstates-1):
                action_v = r[nstates-2] + gamma*previous_value[nstates-2]
            elif state in (4, 5):
#                action_v =  r[succ(state,action)] + \
#                            gamma*(slip*previous_value[succ(state,1-action)] + (1-slip)*previous_value[succ(state,action)])
                action_v = slip*    (r[succ(state,1-action)] + gamma*previous_value[succ(state,1-action)]) + \
                          (1-slip)* (r[succ(state,action)]   + gamma*previous_value[succ(state,action)])
            else:    
                action_v = r[succ(state,action)] + gamma*previous_value[succ(state,action)]
            total_v.append(action_v)
        value[state] = max(total_v)
    if np.mean(np.abs(value-previous_value))<threshold:
        converge = True
        stop = True
    elif iteration>max_iteration:
        stop = True
    else:
        pass
#    if iteration == halfway:
#        plot_value(1, 0, value)
    
#%% show what you did
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
plot_value(1, 1, value)
print(value)