#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:11:57 2018

@author: tom verguts
estimate a value function using dynamic programming
in particular, equation (3.14) from S&B
some states are slippery with probability slip
this is for Q-values in lineworld
"""
# %% preliminaries
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress = True)

def succ(state_pass = 1, action_pass = 1): # successor function
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
halfway = 5 # intermediate-step value matrix to be printed
    
#%% start to iterate
iteration = 0
action_prob = 1/2 # random policy
                
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
    else:
        pass
    if iteration == halfway:
        #plot_value(1, 0, value)
        pass
    
#%% show what you did
print("n iterations = {0}; stopping criterion was{1}reached".format(iteration, [" not ", " "][converge]))
print(Q_value)
fig, axs = plt.subplots(1,3)
centerx, centery, length = 0.5, 0.5, 0.4
resolution = 100
states = "ABCDEFG"
for plot_loop in range(3):
    for loop in range(resolution):
        angle = 1-loop/resolution*2*np.pi
        axs[plot_loop].scatter( centerx + length*np.cos(angle), centery + length*np.sin(angle), s = 0.1, c = [[0.1, 0.01, 0.01]] )                
    for loop in range(len(r)):
        angle = (1-loop/len(r))*2*np.pi
        axs[plot_loop].set_axis_off()
        if plot_loop == 0:
            to_plot = states[loop]
        elif plot_loop == 1:
            to_plot = "{}".format(r[loop])
        else:
            to_plot = "{:.2f} // {:.2f}".format(Q_value[loop, 0], Q_value[loop, 1])
        fig.subplots_adjust(hspace=0.7)
        axs[plot_loop].text(centerx + length*np.cos(angle), centery + length*np.sin(angle), to_plot)
