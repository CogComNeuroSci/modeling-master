#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:29:27 2019

@author: tom verguts
goodness of recovery study for n-armed bandit model RW parameters
for generating table 6.2 in MCP book
note: gradient-based algorithms don't work if you use the non-robust likelihood formulation logL_learn
in this case, use nelder-mead or powell (as i did for generating the tables)
gradient-based algorithms do work if you use the more robust likelihood formulation logL_learnR

nsim = how often do you want to carry out the generate-and-estimate cycle; 
higher means more accurate estimation of mean and standard error for this particular number of trials
note that the standard deviation of the nsim estimates approximates the standard error of the estimator 

prior = (*, 0) means likelihood optimization; if prior[1] > 0, a  bayesian posterior is optimized

data are read from and results written to the current directory
"""
#%% import and initialize
import numpy as np
from numpy import save
from ch6_generation import generate_learn
from ch6_estimation import estimate_learnR

learning_rate, temperature = 0.6, 1

ntrials = [50, 100]
n_sim = 5 
#algorithm = "Powell" # non gradient based
algorithm = "CG"      # gradient based
data_filename = "simulation_data.csv"
extra_label = algorithm + "_bayes_small" # extra label you can give to the output file name 
results_filename = "simulation_results_" + str(n_sim) + "_" + extra_label
est_par = np.ndarray((n_sim, len(ntrials), 4)) # last dim: slice 0 = learn, slice 1 : temp, slice 2 : func_val, slice 3 : iterations
verbose = False

#%% generate data (generate_learn) and estimate parameters (estimate_learn)
for n_loop in range(len(ntrials)):
    for sim_loop in range(n_sim):
        print("trial loop: {:.0%}, simulation loop: {:.0%}".format(n_loop/len(ntrials), sim_loop/n_sim))
        generate_learn(alpha = learning_rate, beta = temperature, ntrials = ntrials[n_loop], \
                       file_name = data_filename, switch = False)
        res = estimate_learnR(nstim = 4, maxiter = 10_000, file_name = data_filename, algorithm = algorithm, prior = (0, 0))
        est_par[sim_loop, n_loop, 0] = res.x[0]
        est_par[sim_loop, n_loop, 1] = res.x[1]
        est_par[sim_loop, n_loop, 2] = res.fun
        est_par[sim_loop, n_loop, 3] = res.success*1
        if verbose:
            print(res)
#%% the result
variables = ["learning rate", "temperature"]

for var_loop, variable in enumerate(variables):
    print("\n subsequent numbers are estimates for number of trials resp. equal to {}".format(ntrials))
    par_str = ""
    for y in range(est_par.shape[1]):
        par_str = par_str + "{:.2f} ".format(np.mean(est_par[:, y, var_loop]))
    print("mean " + variable + " = " + par_str)

    std_str = ""
    for y in range(est_par.shape[1]):
        std_str = std_str + "{:.2f} ".format(np.std(est_par[:, y, var_loop]))
    print("standard error " + variable + " = " + std_str)
    
#%% wrap up
save(results_filename, est_par)