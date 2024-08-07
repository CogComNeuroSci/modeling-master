#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:49:37 2021

@author: tom verguts
example chi-2; see discussion around equation (7.1)
if the model is correct (which it is in this case; data is N independent coin tosses),
then the histogram should correspond to the density plot.
"""

#%% import and initialisations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

nsims, N, p, df = 100000, 100, 0.8, 1
chi2_sim = np.ndarray(nsims)
x = np.linspace(chi2.ppf(0.2, df), chi2.ppf(0.99, df), )

#%% simulation
for sim in range(nsims):
	data = np.random.rand(N)<p  # do N coin tosses
	p_est = np.sum(data)/N      # estimate its probability; with this value, you get good correspondence btw hist and density
	#p_est = p_est-0.1           # use this wrong estimate to obtain discrepancy btw hist and density
	chi2_sim[sim] = N*( ((p_est-p)**2/p) +  (1-p_est-(1-p))**2/(1-p) ) # evaluate the model

#%% show data
plt.hist(chi2_sim, density = True, bins = 20)
plt.plot(x, chi2.pdf(x, df) )