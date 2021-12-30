#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:32:55 2018

@author: tom verguts
this file contains functions that call the optimize function from scipy, 
which can optimze any (well-behaved) function
here, it is applied to the alpha-beta, and two variants of rescorla-wagner model
"""

import numpy as np
import ch6_likelihood as Lik
from scipy import optimize

def estimate_ab(nstim = None, file_name = "", maxiter = 100, algorithm = "Powell"):
    """ estimate parameters of the alpha-beta model.
    by default, the non-gradient-based powell algorithm is used"""
    res = optimize.minimize(Lik.logL_ab, x0 = np.random.rand(2), method = algorithm, tol = 1e-10, args =(nstim,file_name) )
    return res

def estimate_learn(nstim = 4, file_name = "", maxiter = 100, algorithm = "Powell", data = None, prior = (0, 0)):
    """ estimate parameters of the rescorla-wagner model.
    by default, the non-gradient-based powell algorithm is used"""
    res = optimize.minimize(Lik.logL_learn, x0 = np.random.rand(2), method = algorithm, \
                            tol = 1e-10, args = (nstim, file_name, data, prior), options = {"maxiter": maxiter, "disp": True} )
    return res

def estimate_learn2(nstim = 4, file_name = "", maxiter = 100, algorithm = "Powell", data = None, prior = (0, 0)):
    """ estimate parameters of the rescorla-wagner model with two learning rates.
    by default, the non-gradient-based powell algorithm is used"""
    res = optimize.minimize(Lik.logL_learn2, x0 = np.random.rand(3), method = algorithm, \
                            tol = 1e-10, args = (nstim, file_name, data, prior), options = {"maxiter": maxiter, "disp": True} )
    return res