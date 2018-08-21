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
        return np.tanh(netinput)


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
        [weights_suboptimal.append(round(random.uniform(-1, 1), 2))
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


def weight_change():

    return 0


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

    :param tolerated_error:
    How much error between the actual output pattern and the desired output pattern can you handle?
    An example with a tolerated error of .05:
    The altering of the weight matrix will stop as soon as the difference between the activations of the
    actual output pattern and the desired output pattern is smaller than or equal to .05
    So, when the actual pattern is:
        [.95 .95 .03 .03]
    And the desired output pattern is:
        [.99 .99 .01. 01]
    Then the condition is met, and the weight matrix will not be altered any further

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

    weight_question, linear_or_logistic, alpha = asking_questions()

    if weight_question.lower() == 'zero':
        weight_matrix = initialise_weights(input_pattern, output_pattern, zeros=True)
    elif weight_question.lower() == 'random':
        weight_matrix = initialise_weights(input_pattern, output_pattern)
    else:
        raise ValueError('Use correct arguments for the first question.')

    print('\nOriginal weight matrix:')
    for num in range(len(weight_matrix)):
        print('weights from all input units to output unit no. %d: ' % (num+1), weight_matrix[num])

    original_weight_matrix = np.copy(weight_matrix)

    print('\ncycling started\n')

    """
    for cycles in range(loops):
        change_in_weights = weight_change(alpha, input_pattern, output_pattern, weight_matrix,
                                          internal_input(input_pattern, weight_matrix))
        if print_loops:
            print('The adjusted weight matrix: ', change_in_weights)
        weight_matrix = change_in_weights
    """

    return original_weight_matrix, weight_matrix


def loop_delta_until_found(input_pattern, output_pattern, tolerated_error):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    Input patterns are usually numpy arrays

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    Output patterns are usually numpy arrays

    :param tolerated_error:
    How much error between the actual output pattern and the desired output pattern can you handle?
    An example with a tolerated error of .05:
    The altering of the weight matrix will stop as soon as the difference between the activations of the
    actual output pattern and the desired output pattern is smaller than or equal to .05
    So, when the actual pattern is:
        [.95 .95 .03 .03]
    And the desired output pattern is:
        [.99 .99 .01. 01]
    Then the condition is met, and the weight matrix will not be altered any further

    :return:
    Changes the weight matrix until the condition is met (minimal error reached)
    In no consensus is reached, the looping stops after 250 000 cycles to prevent RunTimeErrors from occurring
    """

    cycles = 0
    max_cycles = 500000
    configuration_found = False

    weight_matrix = initialise_weights(input_pattern, output_pattern)
    print('Original (random) weight matrix: \n', weight_matrix)

    int_input = internal_input(input_pattern, weight_matrix)
    alpha = float(input('\nDefine alpha (determines how large the weight change each trial will be): '))
    permission = str(input('Show the weight changes for each cycle (y/n)? '))

    while not configuration_found:
        if permission.lower() in ['yes', 'y']:
            weight_matrix = weight_change(alpha, input_pattern, output_pattern, weight_matrix, int_input)
            print('Cycle %d: adjusted weight matrix: ' % (cycles + 1), weight_matrix)
        else:
            weight_matrix = weight_change(alpha, input_pattern, output_pattern, weight_matrix, int_input)

        new_internal_activation = np.array(internal_input(input_pattern, weight_matrix))
        checking_setup = abs(np.array(output_pattern) - new_internal_activation)

        if all(elements <= tolerated_error for elements in checking_setup):
            print('\nLearning completed (cycle: %d):\n' % cycles,
                  'Original input pattern: ', input_pattern, '\n'
                  ' Original output pattern: ', output_pattern, '\n'
                  ' Adjusted weight matrix: ', weight_matrix, '\n'
                  ' Resulting output pattern: ', new_internal_activation, '\n')
            break

        cycles += 1

        if cycles == max_cycles:
            print('\nNothing found, even after %d cycles ...\n' % max_cycles,
                  'Original input pattern: ', input_pattern, '\n'
                  ' Original output pattern: ', output_pattern, '\n'
                  ' Adjusted weight matrix: ', weight_matrix, '\n'
                  ' Resulting output pattern: ', new_internal_activation, '\n'
                  '\n-- Terminated search --')
            break

    return weight_matrix
