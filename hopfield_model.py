#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:15:46 2017

@author: tom
Hopfield model
"""

# import
import numpy as np
from numpy import random

# initialize
n_trials = 1
w_p = 1
w_m = -1
n_unit = 6
max_n_step = 30
threshold = np.ones(n_unit)/2
stop_threshold = 0.5 
weight = np.zeros((6,6))
for loop in range(n_unit):
    weight[loop,loop] = w_p
weight[1,:1] = [w_p]
weight[2,:2] = [w_p, w_p]
weight[3,:3] = [w_m, w_m, w_m]
weight[4,:4] = [w_m, w_m, w_m, w_p]
weight[5,:5] = [w_m, w_m, w_m, w_p, w_p]

for loop in range(n_trials):
    # start
    x = random.randint(low=0,high=2,size=n_unit) # random starting pattern
    print("start:{}".format(x))
    counter = 0
    stop_crit = False
    while not stop_crit and counter<max_n_step:
        k = random.randint(n_unit)
        x_new = np.array(np.dot(weight,x)>threshold, dtype=int)
        deviance = np.sum(np.abs(x-x_new))
        if deviance<stop_threshold:
            stop_crit = True
        counter += 1
        x = x_new
        print(x)
if stop_crit:
    crit_string = ""
else:
    crit_String = "not "
print("stop criterion " + crit_string + "reached")