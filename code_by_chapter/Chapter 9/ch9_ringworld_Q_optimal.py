#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate a value function using dynamic programming
some states are slippery with probability slip
this program calculates the optimal-policy Q values in a circular ringworld
"""
# %% import and initialize
import numpy as np
import matplotlib.pyplot as plt
from ch9_plotting import plot_value_circular

np.set_printoptions(precision=4, suppress = True)

def succ(state_pass = 1, action_pass = 1):
    """the successor function"""
    if (state_pass == 0) and (action_pass == 0):
        return nstates - 1
    elif (state_pass == nstates - 1) and (action_pass == 1):
        return 0
    else:
        return state_pass + action_pass*2 - 1
    
nstates, nactions = 7, 2
r = [0, 0, 0.7, 0, 0, 0, 0.8]
slip = 0.9
slip_states = [0]
Q_value = np.random.random((nstates, 2))
gamma = 0.8 # discount factor
stop, converge, threshold, max_iteration = False, False, 0.01, 10000

    
#%% main code
# start to iterate
iteration = 0                
while stop == False:
    previous_Q_value = np.copy(Q_value)
    iteration += 1
    for state in range(nstates):
        for action in range(nactions):
            if state in slip_states:
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
    
#%% print and plot results
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
print(Q_value)
fig, axs = plt.subplots(1,3)
centerx, centery, length = 0.5, 0.5, 0.4
resolution = 100
states = "ABCDEFG"
fig.suptitle("values in ringworld")
for plot_loop in range(3):
    plot_value_circular(fig, axs, plot_loop, Q_value, r, centerx, centery, length, resolution = resolution, states = states)
