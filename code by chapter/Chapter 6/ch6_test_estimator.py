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
"""

#%% import and initialize
import numpy as np
import ch6_generation as generator
import ch6_estimation as estimator

model = "learn" # alpha-beta model or learning model
if model == "ab":
    nstim = None
else:   # learning model
    nstim = 4    
    
alpha_list = [0.2] # in learning model, this is learning rate
beta_list =  [1.5] # in learning mdodel, this is slope (inv temperature)
param_list = []
file_name_to_write = "simulation_data_1.csv"
np.set_printoptions(precision = 2, suppress = True)

#%% define main function
def main():
    for alpha_loop in alpha_list:
        for beta_loop in beta_list:
            print("parameters are {} and {}".format(alpha_loop, beta_loop))
            if model == "ab":              
                # generate data and send them to a file
                generator.generate_ab(alpha = alpha_loop, beta = beta_loop, ntrials = 500, file_name = file_name_to_write) 
                # import data from a file and estimate its parameters
                pars = estimator.estimate_ab(nstim, file_name_to_write)
            else:
                # generate data and send them to a file
                generator.generate_learn(alpha = alpha_loop, beta = beta_loop, ntrials = 1000, file_name = file_name_to_write)
                # import data from a file and estimate its parameters
                pars = estimator.estimate_learn(nstim, file_name_to_write)
            param_list.append([alpha_loop, beta_loop, pars[0], pars[1]])     
    c = np.corrcoef(param_list, rowvar = False)        
    print("corr btw real and estimated parameters equals {}".format(c))

#%% main code
main()