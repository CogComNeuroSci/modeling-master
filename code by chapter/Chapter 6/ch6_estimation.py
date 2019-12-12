#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:32:55 2018

@author: tom verguts
this calls the optimize function from scipy, 
which can optimze any (well-behaved) function
"""

import numpy as np
import ch6_ST_likelihood as Lik
from scipy import optimize

def estimate_ab(nstim = None, file_name = "simulation_data.csv"):
    estim_param = optimize.fmin(Lik.logL_ab, np.random.rand(2), args =(nstim,file_name), maxiter = 100, ftol = 0.001)
    return estim_param

def estimate_learn(nstim = 4, file_name = "simulation_data.csv"):
    estim_param = optimize.fmin(Lik.logL_learn, np.random.rand(2), args =(nstim,file_name), maxiter = 10, ftol = 0.001)
    return estim_param