#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:57:23 2018

@author: tom
test an estimator
"""

import numpy as np
import ch6_DM_generation as generator
import ch6_DM_estimation as estimator

# initialize
alpha_list = [0.3, 0.6]
beta_list =  [0.3, 0.5, 0.7]
param_list = []
nstim = 4
file_name_to_write = "simulation_data_1.csv"

#def logit(beta_in,x1,x2):
#    return 1/(1+np.exp(beta_in*(x2-x1)))

def main():
    for alpha_loop in alpha_list:
        for beta_loop in beta_list:
            print([alpha_loop, beta_loop])
            generator.generate_ab(alpha = alpha_loop, beta = beta_loop, ntrials = 500, file_name = file_name_to_write)
            pars = estimator.estimate_ab(nstim, file_name_to_write)
            param_list.append([alpha_loop, beta_loop, pars[0], pars[1]])     
    print(np.corrcoef(param_list, rowvar = False))

main()