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
    This parameter can have to values:
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
    Determines which type of activation function is used
    This parameter can have to values:


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

    np.set_printoptions(suppress=True)
    weights = np.array(weights)

    for i in range(len(output_pattern)):
        altered_weights = weights[i]
        for j in range(len(altered_weights)):
            internal_activation = internal_input(input_pattern, weights, act_function=function_word)[0]
            delta = np.array(output_pattern[i]) - internal_activation[i]
            altered_weights[j] = alpha * activation_function(input_pattern[j]) * delta
            if abs(delta) <= .1:
                    break
        weights[i] = np.round(altered_weights, 3)
    return weights


def asking_questions():

    while True:
        answer_one = str(input('Weights all set to zero? Type: zero\n'
                               'Weights all determined randomly? Type: random\n'
                               'Answer: '))
        if answer_one.lower() in ['zero', 'random']:
            break
        else:
            print('\n! ! !\nUnexpected input. \nPlease try again.\n! ! !\n')
            continue

    while True:
        answer_two = str(input('\nWill you use a linear activation function to transform netinput? Type: linear\n'
                               'Will you use a logistic activation function to transform netinput? Type: logistic\n'
                               'Answer: '
                               ''))
        if answer_two in ['linear', 'logistic']:
            break
        else:
            print('\n! ! !\nUnexpected input. \nPlease try again.\n! ! !\n')
            continue

    while True:
        try:
            answer_three = float(input('\nDetermine the value for parameter alpha (the stepsize)\n'
                                       'This value has to be a float\n'
                                       'Answer: '))
            break
        except ValueError:
            print('\n! ! !\nUnexpected input. \nPlease try again.\n! ! !\n')
            continue

    return answer_one, answer_two, answer_three


def loop_delta(input_pattern, output_pattern, loops=50, print_loops=True):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    Input patterns are usually numpy arrays

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    Output patterns are usually numpy arrays

    :param loops:
    The number cycles where the weight matrix is altered to close the gap between the actual output
    and the desired output

    :param print_loops:
    Defines whether the alterations in the weight matrices are printed or not
    Two values are possible:
        True makes sure that all the changes in the weight matrix are printed
        False means that this doesn't happen

    :return:
    Changes the weight matrix for a fixed amount of cycles, or until the condition is met (minimal error reached)
    """

    random_zero, linear_logistic, alpha = asking_questions()

    if random_zero == 'zero':
        weights = initialise_weights(input_pattern, output_pattern, zeros=True)
    else:
        weights = initialise_weights(input_pattern, output_pattern, zeros=False)

    for i in range(loops):
        all_met = False
        resulting_matrix = weight_change(alpha, input_pattern, output_pattern, weights, function_word=linear_logistic)
        if print_loops:
            print("Altered matrix: \n", resulting_matrix)

        for j in range(len(output_pattern)):
            if output_pattern[j] != np.sum(np.multiply(input_pattern, resulting_matrix[j])):
                break
            all_met = True

        if all_met:
            print('\nLearning completed after completing loop nr %d\n' % (i+1),
                  '\nAll loops executed, and solution yielded:')
            for l in range(len(resulting_matrix)):
                print('Netinput for unit %d: ' % (l + 1), np.sum(np.multiply(input_pattern, weights[l])),
                      '\nDesired: ', output_pattern[l], '\n')
            print('Adjusted weight matrix:')
            return resulting_matrix
        else:
            weights = resulting_matrix

    print('\nNothing satisfactory found\n'
          'All loops executed to no avail:')
    for l in range(len(weights)):
        print('Netinput for unit %d: ' % (l+1), np.sum(np.multiply(input_pattern, weights[l])),
              '\nDesired: ', output_pattern[l], '\n')
    print('\nAdjusted weight matrix:')
    return weights
