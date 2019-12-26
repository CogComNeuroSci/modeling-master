#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:53:58 2018

@author: tom verguts
data generator for two models; more can be added
"""

import numpy as np
import pandas as pd
import random

def logit(beta_in,x1,x2):
    return 1/(1+np.exp(beta_in*(x2-x1)))

# generate data for the alpha-beta (aka, log-linear) model
def generate_ab(alpha = 0.5, beta = 0.3, ntrials = 100, file_name = "simulation_data.csv"):
    column_list = ["difficulty", "accuracy"]
    data = pd.DataFrame(columns=column_list)
    # simulate data
    for loop in range(ntrials):
        # choose stimulus difficulty X
        X = (random.random()<0.5)*1 # *1 to change True to 1 and False to 0
        p1 = alpha*(beta**X)
        choice = (random.random()<p1)*1 # was subject successful or not
        data.loc[loop] = [X, choice]
    # write data to file
    data.to_csv(file_name, columns = column_list)

# generate data for the learning model
def generate_learn(w0 = 0, alpha = 0.5, beta = 0.2, ntrials = 1000, nstim = 4, file_name = "simulation_data.csv", switch = False):    
    # initialize
    prob = [0.1, 0.3, 0.6, 0.8] # probabilities of reward for each stim
    column_list = ["stim1", "stim2", "choice", "Reward"]
    data = pd.DataFrame(columns=column_list)
    value = w0*np.ones(nstim)
    # simulate data
    for loop in range(ntrials):
        if switch:
            if (loop%50) == 0:
                prob.reverse()
        # choose 2 stimuli out of nstim
        stim = random.sample(range(nstim),2)    
        p0 = logit(beta,value[stim[0]],value[stim[1]])
        choice = int(random.random()>p0)
        Reward = int(random.random()<prob[stim[choice]])
        value[stim[choice]] = value[stim[choice]] + alpha*(Reward-value[stim[choice]]) # Rescorla-Wagner update 
        data.loc[loop] = [stim[0], stim[1], choice, Reward]
    # write data to file
    data.to_csv(file_name, columns = column_list)