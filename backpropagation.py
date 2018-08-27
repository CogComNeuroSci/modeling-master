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
        return np.round(.5 * ((np.e + 1/np.e) / (np.e - 1/np.e)) * np.tanh(netinput), 5)


def initialise_weights(input_pattern, output_pattern, zeros=False):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    Input patterns are usually numpy arrays

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    Output patterns are usually numpy arrays

    :param zeros:
    Defines how you want the weights to be set:
    This parameter can have two values:
        True means that all weights will be set to zero
        False means that all weights will be determined randomly by sampling from the following interval:
                [-1, 1]

    :return:
    This function returns a random weight matrix
    In this case, the weights are random floats (rounded to two decimal places) in the following range:
    [-5, 5]
    The weight matrix is an array of n x m long, where n is the length of the input pattern,
    and m the length of the output pattern
    The array has m sub arrays, each with n elements in it
    This signifies the weights from all n input units to output unit m
    So, the first sub array represents the weights from all input units to the first output unit
    """

    number_of_weights = len(input_pattern) * len(output_pattern)

    weights_suboptimal = []
    weights = []

    if zeros:
        [weights_suboptimal.append(0)
         for i in range(number_of_weights)]
        [weights.append(weights_suboptimal[j:j + len(input_pattern)])
         for j in range(0, len(weights_suboptimal), len(input_pattern))]
    else:
        [weights_suboptimal.append(round(random.uniform(-3, 3), 2))
         for i in range(number_of_weights)]
        [weights.append(weights_suboptimal[j:j + len(input_pattern)])
         for j in range(0, len(weights_suboptimal), len(input_pattern))]

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


def weight_change(alpha, input_pattern, output_pattern, weights, function_word='logistic'):

    """
    :param alpha:
    The stepsize
    The larger this parameter is, the more drastic the weight changes in each trial will be

    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    Input patterns are usually numpy arrays

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    Output patterns are usually numpy arrays

    :param weights:
    The weight matrix for our input-, and output pattern.
    This matrix can be initialised using the function 'initialise_weights()'.

    :param function_word:
    Determines the type of the activation function that is used
    This parameter can have two values:
        'linear' means that a linear activation function is used to "transform" the netinput
        'logistic' means that a logistic activation function is used to transform the netinput

    :return:
    Determines the weight change for each trial based on the internal input
    The difference the desired activation level and the actual activation level is used to do so
    """

    np.set_printoptions(suppress=True)
    weights = np.array(weights)

    for i in range(len(output_pattern)):
        altered_weights = weights[i]
        for j in range(len(altered_weights)):
            internal_activation = internal_input(input_pattern, weights, act_function=function_word)[0]
            delta = activation_function(np.array(output_pattern[i])) - internal_activation[i]
            altered_weights[j] = altered_weights[j] + \
                alpha * activation_function(input_pattern[j]) * delta
            if abs(delta) <= .1:
                    break
        weights[i] = np.round(altered_weights, 3)
    return weights

