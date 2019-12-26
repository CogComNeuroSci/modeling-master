#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:32:55 2018

@author: tom verguts
this calls the optimize function from scipy, 
which can optimze any (well-behaved) function
"""

import numpy as np
import ch6_likelihood as Lik
from scipy import optimize

def estimate_ab(nstim = None, file_name = "simulation_data.csv", maxiter = 10, algorithm = "Powell"):
    res = optimize.minimize(Lik.logL_ab, x0 = np.random.rand(2), method = algorithm, tol = 1e-10, args =(nstim,file_name) )
    return res

def estimate_learn(nstim = 4, file_name = "simulation_data.csv", maxiter = 10, algorithm = "Powell"):
    res = optimize.minimize(Lik.logL_learn, x0 = np.random.rand(2), method = algorithm, \
                            tol = 1e-10, args =(nstim, file_name, (0, 5)), options = {"maxiter": maxiter, "disp": True} )
    return res