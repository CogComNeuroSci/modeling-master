#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:57:23 2018

@author: tom verguts
the set of py files ch6_*[test_estimator, generation, estimation, likelihood]
can be used to estimate parameters from a model, and check whether it's possible
at all to recover parameters from the model
the current program (test_estimator) can be used to test if you can recover (ie, accurately estimate)
parameters from a model
Note: output of scipy.optimize used to be an array of parameter values... now it's a bigger object with .x being the estimates
number of rows in param_list equals len(alpha_list)*len(beta_list)
"""

#%% import and initialize
import numpy as np
import ch6_generation as generator
import ch6_estimation as estimator # this function itself uses ch6_likelihood

model = "learn" # alpha-beta model or learning model
if model == "ab":
    nstim = None
else:   # learning model
    nstim = 4    

alpha_list = [0.3, 0.7] #  in ab model, this is overall difficulty   ; in learning model, this is learning rate
beta_list =  [0.3, 0.5] #  in ab model, this is hard-trial difficulty; in learning model, this is slope (inv temperature)
param_list = []              #  list of real and estimated parameters (one column for each)
np.set_printoptions(precision = 2, suppress = True)


def main(algorithm_used: str = "powell", n_trials: int = 100, file_name_to_write: str = "simulation_data_1.csv"):
    """define main function.
    User can choose estimation algorithm (algorithm_used), n trials (n_trials), and file name to write to (file_name_to_write)"""""
    for alpha_loop in alpha_list:
        for beta_loop in beta_list:
            print("parameters are {} and {}".format(alpha_loop, beta_loop))
            if model == "ab":              
                # generate data and send them to a file
                generator.generate_ab(alpha = alpha_loop, beta = beta_loop, ntrials = n_trials, file_name = file_name_to_write) 
                # import data from a file and estimate its parameters
                pars = estimator.estimate_ab(nstim, file_name_to_write, algorithm = algorithm_used)
            else:
                # generate data and send them to a file
                generator.generate_learn(alpha = alpha_loop, beta = beta_loop, ntrials = n_trials, file_name = file_name_to_write)
                # import data from a file and estimate its parameters
                pars = estimator.estimate_learn(nstim, file_name_to_write, algorithm = algorithm_used)
            param_list.append([alpha_loop, beta_loop, pars.x[0], pars.x[1]])     
    c = np.corrcoef(param_list, rowvar = False)        
    if len(param_list)>1:
        print("corr btw real and estimated parameters equals {}".format(c))
    else:
        print("single parameter, cannot calculate correlation")

#%% main code
main(algorithm_used = "powell", n_trials = 1000, file_name_to_write = "simulation_data_1.csv")