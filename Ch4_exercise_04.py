#!/usr/bin/python3

import Ch0_delta_learning as delta_learning
import numpy as np
import os

"""
@author: Pieter Huycke
email: pieter.huycke@ugent.be
"""

os.chdir(r'C:\Users\Pieter\Downloads\Modeling\code\modeling-master')


def auto_association(input_pattern, weight_vars=(False, False), loops=10000, alpha=1.5, allowed_error=.01,
                     verbose=True):

    # starting weights
    original_weights = delta_learning.initialise_weights(input_pattern, input_pattern,
                                                         zeros=weight_vars[0], predefined=weight_vars[1])
    altered_weights = np.copy(original_weights)

    # testing the input with the original weight matrix
    seeing_pattern = delta_learning.internal_input(input_pattern, altered_weights)[0]

    if verbose:
        print('\nSeeing the pattern with a random weight matrix:')
        print(list(np.round(seeing_pattern, 2)))
        print('Desired output:')
        print(input_pattern)
    else:
        pass

    print('Altering weight matrix for input...')
    error_margin = allowed_error
    tracking_altered_activations = np.linspace(-1., 1., len(input_pattern))

    for j in range(loops):
        new_weights = delta_learning.weight_change(alpha, input_pattern, input_pattern, altered_weights,
                                                   function_word='logistic')
        # testing new weights
        activations = delta_learning.internal_input(input_pattern, new_weights, act_function='logistic')[0]
        test = np.array(input_pattern) - np.array(activations)
        if all(abs(elements) < error_margin for elements in test):
            print('Satisfying weight matrix found after cycle number %d!' % (j + 1))
            break
        if np.array_equal(test, tracking_altered_activations):
            print('Steady state reached in cycle number %d!' % (j + 1))
            break
        altered_weights = new_weights
        tracking_altered_activations = test

    # testing the input with new weights
    seeing_pattern = delta_learning.internal_input(input_pattern, altered_weights, act_function='logistic')[0]

    if verbose:
        print('\nActivation after auto association:')
        print(list(np.round(seeing_pattern, 2)))
        print('Desired output:')
        print(list(np.array(input_pattern)))
    else:
        pass
    print('- - -')

    return altered_weights


"""
new_weight_matrix = auto_association([.99, .01, .99, .99, .01, .01, .01, .01, .99, .99, .99, .99, .99, .01, .01, .01],
                                     weight_vars=(False, False), loops=10000, alpha=1.5, allowed_error=.01)
"""

# read data
all_patterns = np.loadtxt('patterns_different_dogs.txt')

# create weight matrix based on the first input pattern
first_pattern = all_patterns[0]
fake_weight_matrix = delta_learning.initialise_weights(first_pattern, first_pattern, zeros=True, predefined=False)

for i in range(50):

    print('Now working on cycle number %d...' % (i+1))
    auto_association_matrix = auto_association(all_patterns[i], weight_vars=(False, False),
                                               loops=10000, alpha=1.5, allowed_error=.01, verbose=False)
    fake_weight_matrix += auto_association_matrix

altered_matrix = fake_weight_matrix

