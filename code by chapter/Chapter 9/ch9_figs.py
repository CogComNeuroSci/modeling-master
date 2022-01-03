#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
plot figure 9.2 in MCP book
estimate a value function using dynamic programming
currently implemented for random policy only
in particular, equation (3.14) for V estimates, from Sutton & Barto RL book
all transition probs p() are deterministic in this case
actions are (in this order): up, down, left, right
"""
#%% import and initialize
import numpy as np
import matplotlib.pyplot as plt
from ch9_plotting import plot_value

np.set_printoptions(precision=4, suppress = True)

def state2rc(state_pass = 1):  
    """function to transform from state number to (row, column) pair"""
    return state_pass // 5, state_pass % 5

def succ(state_pass = 1, action_pass = 1): 
	"""successor function: sutton & barto call this p(s',r / s,a),
	but note that it is deterministic in this case so it can be represented more
	simply than with a probability distribution"""
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
    

nstates = 25
value = np.random.random((5,5))
ntrials = 100
gamma = 0.9 # temporal discount parameter
stop, converge, threshold, max_iteration = False, False, 0.01, 20
halfway = 5 # intermediate-step value matrix to be printed

fig, axs = plt.subplots(1, 3)
    
# %% Dynamic programming algorithm
# initial state
plot_value(fig, axs, 0, 0, value, title = "initial")
print("initial value estimates for the random policy: \n")
print(value)

# start to iterate
iteration = 0
while stop == False:
    previous_value = np.copy(value)
    iteration += 1
    for state in range(nstates):
        row, column = state2rc(state)
        total_v = 0
        for action in range(4):  # sum across a in eq (3.14)
            action_prob = 1/4    # random policy has this pi(a/s)
            if (row==0) & (column==1):   # fly from A to A' (state 21)
                action_v = 10+gamma*previous_value[state2rc(21)]
            elif (row==0) & (column==3): # fly from B to B' (state 13)
                action_v = 5+gamma*previous_value[state2rc(13)]
            elif (column==0) & (action==3): # bump into left wall
                action_v = -1+gamma*previous_value[state2rc(state)]
            elif (column==4) & (action==1): # bump into right wall
                action_v = -1+gamma*previous_value[state2rc(state)]
            elif (row==0) & (action==0):    # bump into upper wall
                action_v = -1+gamma*previous_value[state2rc(state)]
            elif (row==4) & (action==2):    # bump into lower wall
                action_v = -1+gamma*previous_value[state2rc(state)]
            else:                           # move in the grid
                action_v = 0+gamma*previous_value[succ(state,action)]
            action_v *= action_prob # multiply with pi(a/s) (here constant, so in principle could go out of the summation)
            total_v += action_v     # sum across actions a
        value[row,column] = total_v
    if np.mean(np.abs(value-previous_value))<threshold:
        converge = stop = True
    elif iteration>max_iteration:
        stop = True
    if iteration == halfway:
        plot_value(fig, axs, 0, 1, value, title = "halfway")
    
#%% show final results of the iteration process
plot_value(fig, axs, 0, 2, value, title = "final")
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
print("final value estimates for the random policy: \n")
print(value)