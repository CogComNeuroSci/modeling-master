#!/usr/bin/python3

import Ch0_delta_learning as delta_learning
import numpy as np

"""
@author: Pieter Huycke
email: pieter.huycke@ugent.be
"""

np.set_printoptions(suppress=True)

"""
help(delta_learning)
to learn more about the entire module

help(delta_learning.loop_delta)
to learn more about one of the specific functions
"""

inputted_pattern = [.99, .01, .99, .01, .99, .01]
desired_output = [.99, .99, .01, .01]

best_weights = delta_learning.loop_delta(inputted_pattern, desired_output,
                                         loops=1000, print_loops=False, error_margin=.01)

