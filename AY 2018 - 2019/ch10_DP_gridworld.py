#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate a value function using dynamic programming
in particular, equation (3.14) from S&B
note: all p() are deterministic in this case
"""
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress = True)

# %% figure 10.2
def state2rc(state_pass = 1):  # state to (row, column)
    return state_pass // 5, state_pass % 5

def succ(state_pass = 1, action_pass = 1): # successor function
    row, column = state2rc(state_pass)
    if action_pass == 0:
        row -= 1
    elif action_pass == 1:
        column += 1
    elif action_pass == 2:
        row += 1
    else:
        column -= 1
    return row, column    
    
def plot_value(row, column, value_matrix):
    offset = 0.05
    for xloop in range(5):
        for yloop in range(5):
            axs[row, column].text(offset + yloop/5, offset + (4 - xloop)/5, "{:.1f}".format(value_matrix[xloop, yloop]))
    axs[row, column].grid(True)
    axs[row, column].set_xticklabels(" ")
    axs[row, column].set_yticklabels(" ")
    return

nstates = 25
value = np.random.random((5,5))
#ntrials = 1000
gamma = 0.95 # discount factor
stop, converge, threshold, max_iteration = False, False, 0.01, 10000
halfway = 5 # intermediate-step value matrix to be printed

fig, axs = plt.subplots(2, 2)
    
# start to iterate
plot_value(0, 1, value)
print(value)
iteration = 0
while stop == False:
    previous_value = value + 0
    iteration += 1
    for state in range(nstates):
        row, column = state2rc(state)
        total_v = 0
        for action in range(4):
            action_prob = 1/4 # random policy
            if (row==0) & (column==1):
                action_v = 10+gamma*previous_value[state2rc(21)]
            elif (row==0) & (column==3):
                action_v = 5+gamma*previous_value[state2rc(13)]
            elif (column==0) & (action==3):
                action_v = -1+gamma*previous_value[state2rc(state)]
            elif (column==4) & (action==1):
                action_v = -1+gamma*previous_value[state2rc(state)]
            elif (row==0) & (action==0):
                action_v = -1+gamma*previous_value[state2rc(state)]
            elif (row==4) & (action==2):
                action_v = -1+gamma*previous_value[state2rc(state)]
            else:
                action_v = 0+gamma*previous_value[succ(state,action)]
            action_v *= action_prob
            total_v += action_v
        value[row,column] = total_v
    if np.mean(np.abs(value-previous_value))<threshold:
        converge = True
        stop = True
    elif iteration>max_iteration:
        stop = True
    else:
        pass
    if iteration == halfway:
        plot_value(1, 0, value)
    
# show what you did
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
plot_value(1, 1, value)
print(value)