#!/usr/bin/python3

import numpy as np
import random


"""
@author: Pieter Huycke
email: pieter.huycke@ugent.be
"""


def activation_function(netinput, form='logistic'):

    """
    :param netinput:
    The netinput for a certain input unit
    This is usually calculated using the linearity principle (see ch04 p01)

    :param form:
    Determines the type of the activation function that is used
    This parameter can have two values:
        'linear' means that a linear activation function is used to "transform" the netinput
        'logistic' means that a logistic activation function is used to transform the netinput

    :return:
    This function simply returns the netinput, as is often the case in linear activation functions
    So, no transformation is done on the netinput
    """

    if form == 'linear':
        return netinput
    else:
        calculated = 1 / (1 + np.exp(-5 * (netinput - .5)))
        return calculated


def make_network(setup=None, zeros=False):

    if setup is None:
        setup = [2, 2, 1]
    size_network = len(setup)
    network = []

    for i in range(size_network):
        sub_network = []
        for j in range(setup[i]):
            random_number = 0
            if not zeros:
                random_number = round(random.uniform(-1, 1), 2)
            sub_network.append(random_number)
        network.append(sub_network)

    return network


def initialise_weights(network, zeros=True):

    index = 1
    weights = []

    while index != len(network):
        suboptimal_weights = []

        input_pattern = network[index - 1]
        output_pattern = network[index]

        number_of_weights = len(input_pattern) * len(output_pattern)
        for i in range(number_of_weights):
            if zeros:
                suboptimal_weights.append(0)
            else:
                suboptimal_weights.append(round(random.uniform(-5, 5), 2))
        weights.append(suboptimal_weights)
        index += 1

    return weights


def internal_input(input_pattern, weights, act_function='logistic'):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    Input patterns are usually numpy arrays

    :param weights:
    The weight matrix for our input-, and output pattern.
    This matrix can be initialised using the function 'initialise_weights()'.

    :param act_function:
    Determines the type of the activation function that is used
    This parameter can have two values:
        'linear' means that a linear activation function is used to "transform" the netinput
        'logistic' means that a logistic activation function is used to transform the netinput

    :return:
    Returns the netinput after it was transformed using a logistic activation function
    The netinput is calculated using the linearity principle (input * weights for all sending units)
    Subsequently, this summed input is transformed
    This function returns an array with all activations for all output units
    """

    activations = []

    for i in range(len(weights)):
        added_activation = np.sum(np.multiply(activation_function(input_pattern), weights[i]))
        activations.append(added_activation)

    return activations, act_function


def chunks(l, n):
    n = max(1, n)
    parts = []
    [parts.append(l[i:i+n]) for i in range(0, len(l), n)]
    return parts


def calculate_output(network, weights):

    activation_levels = []

    for k in range(len(network) - 1):
        subset_network = network[k]
        subset_weights = chunks(weights[k], len(subset_network))
        extra_step = []
        for l in range(len(subset_weights)):
            result = np.round(np.sum(np.multiply(subset_weights[l], subset_network)), 2)
            extra_step.append(np.round(activation_function(result, form='logistic'), 2))
        activation_levels.append(extra_step)

    return activation_levels


test_network = make_network([2, 2, 1], zeros=False)
weight_matrix = initialise_weights(test_network, zeros=False)

print('Network: ', test_network)
print('Weight matrix: ', weight_matrix)
print('Output:', calculate_output(test_network, weight_matrix))
