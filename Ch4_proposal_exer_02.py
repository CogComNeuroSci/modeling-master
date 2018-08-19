#!/usr/bin/python3

import numpy as np
import random


"""
@author: Pieter Huycke
email: pieter.huycke@ugent.be
"""


def logistic_activation(netinput):

    """
    :param netinput:
    The netinput for a certain input unit.
    This is usually calculated using the linearity principle (see ch04 p01).

    :return:
    The value for a certain netinput when using a logistic activation function.
    This value is basically a transformation of the netinput.
    Doing this makes sure that input at a certain input node is always restrained between certain boundaries.
    """

    return 1 / (1 + np.exp(-netinput))


def initialise_weights(input_pattern, output_pattern):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    Input patterns are usually numpy arrays

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    Output patterns are usually numpy arrays

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

    [weights_suboptimal.append(round(random.uniform(-5, 5), 2))
     for i in range(number_of_weights)]
    [weights.append(weights_suboptimal[j:j + len(input_pattern)])
     for j in range(0, len(weights_suboptimal), len(input_pattern))]

    return weights


def internal_input(input_pattern, output_pattern, weights):

    """
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

    :return:
    Returns the netinput after it was transformed using a logistic activation function
    The netinput is calculated using the linearity principle (input * weights for all sending units)
    Subsequently, this summed input is transformed
    This function returns an array with all activations for all output units
    """

    activations = []

    for i in range(len(output_pattern)):
        added_activation = np.sum(np.multiply(input_pattern, weights[i]))
        rounded_activation = np.round(added_activation, 2)
        result = logistic_activation(rounded_activation)
        activations.append(result)

    return activations


def weight_change(alpha, input_pattern, output_pattern, weight_matr, internal_activations):

    """
    :param alpha:
    The step size, this parameter influences how large the weight change is in each cycle

    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    Input patterns are usually numpy arrays

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    Output patterns are usually numpy arrays

    :param weight_matr:
    The original weight matrix that yields a specific activation pattern
    This weight matrix is changed, and often used in later weight changing cycles

    :param internal_activations:
    The activation of the output units as calculated by internal_input()
    This is used to determine the difference between the desired output (output pattern)
    and the actual output (what is yielded by internal_input())

    :return:
    An altered weight matrix that yields a better match between the desired output and the actual output
    """

    for l in range(len(output_pattern)):
        delta = output_pattern[l] - internal_activations[l]
        for m in range(len(input_pattern)):
            weight_matr[l][m] = round(weight_matr[l][m] + (alpha*logistic_activation(weight_matr[l][m]) * delta *
                                                           output_pattern[l] * (1 - output_pattern[l])), 2)

    return weight_matr


def loop_delta(input_pattern, output_pattern, loops, tolerated_error):

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
    Changes the weight matrix for a fixed amount of cycles, or until the condition is met (minimal error reached)
    """

    weight_matrix = initialise_weights(input_pattern, output_pattern)
    print('Original (random) weight matrix: \n', weight_matrix)

    int_input = internal_input(input_pattern, output_pattern, weight_matrix)
    alpha = float(input('\nDefine a constant which influences how large the weight change each trial will be: '))
    permission = str(input('Show the weight changes for each cycle (y/n)? '))

    consensus_reached = False

    for cycles in range(loops):
        if permission.lower() in ['yes', 'y']:
            weight_matrix = weight_change(alpha, input_pattern, output_pattern, weight_matrix, int_input)
            print('Cycle %d: adjusted weight matrix: ' % (cycles + 1), weight_matrix)
        else:
            weight_matrix = weight_change(alpha, input_pattern, output_pattern, weight_matrix, int_input)

        new_internal_activation = np.array(internal_input(input_pattern, output_pattern, weight_matrix))
        checking_setup = abs(np.array(output_pattern) - new_internal_activation)

        if all(elements <= tolerated_error for elements in checking_setup):
            print('\nLearning completed (cycle: %d):\n' % cycles,
                  'Original input pattern: ', input_pattern, '\n'
                  ' Original output pattern: ', output_pattern, '\n'
                  ' Adjusted weight matrix: ', weight_matrix, '\n'
                  ' Resulting output pattern: ', new_internal_activation, '\n')
            consensus_reached = True
            break

    if not consensus_reached:
        print('\nNothing found, even after %d cycles ...\n' % loops,
              'Original input pattern: ', input_pattern, '\n'
              ' Original output pattern: ', output_pattern, '\n'
              ' Adjusted weight matrix: ', weight_matrix, '\n'
              ' Resulting output pattern: ', new_internal_activation, '\n'
              '\n-- Terminated search --')

    return weight_matrix


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

    int_input = internal_input(input_pattern, output_pattern, weight_matrix)
    alpha = float(input('\nDefine alpha (determines how large the weight change each trial will be): '))
    permission = str(input('Show the weight changes for each cycle (y/n)? '))

    while not configuration_found:
        if permission.lower() in ['yes', 'y']:
            weight_matrix = weight_change(alpha, input_pattern, output_pattern, weight_matrix, int_input)
            print('Cycle %d: adjusted weight matrix: ' % (cycles + 1), weight_matrix)
        else:
            weight_matrix = weight_change(alpha, input_pattern, output_pattern, weight_matrix, int_input)

        new_internal_activation = np.array(internal_input(input_pattern, output_pattern, weight_matrix))
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


def main():

    """
    :return:
    The main function to get the whole program starting
    Asks user input to define the way the delta rule is implemented
    """

    print(' -------------- \n'
          '| General idea |\n'
          ' -------------- \n'
          'In this second exercise we take a closer look at the Delta rule.\n'
          'We learned in Ch 04 that the weight change under the Delta rule is proportional to the difference\n'
          'between the "desired" output and the "actual" output: t_i - y_i. \n'
          'Other factors that influence the weight change are alpha (step size), \n'
          'and the activation of the sending input unit: x_j.\n\n'
          'In this exercise, we will investigate how weight change under the Delta rule occurs.\n'
          'To do so, we have to change the weights between the input units and output units to make sure that\n'
          'the resulting weight matrix yields activations for all output units that are all really close to the \n'
          'the desired output values we defined earlier.\n'
          'To calculate the activation based on the netinput, we will use a logistic activation function.\n\n'
          'In this exercise, you are going to use this program to manipulate the available parameter values,\n'
          'and investigate how these manipulations lead to a weight matrix that yields activations close to.\n'
          'the output we desire.\n\n'
          'Specifically, you should study how the following parameters influence the learning process:\n'
          '\t\t1) alpha: the stepsize\n'
          '\t\t2) the form of the input-, and output patterns\n'
          '\t\t3) whether you cycle a fixed amount of times, or you cycle until a solution is found\n')

    escaping = 0

    while True:
        starting_cue = str(input('Would you like to start the program (y/n)? '))
        if starting_cue not in ['yes', 'y', 'no', 'n']:
            print("Unexpected input. Please type 'y' or 'n' only.\n")
            continue
        else:
            if starting_cue in ['y', 'yes']:
                pass
            else:
                print('Terminating program ...')
                escaping = 1
                break
        if escaping == 1:
            break
        while True:
            print('\n'
                  ' ------------------- \n'
                  '| Kicking the tires |\n'
                  ' ------------------- ')
            which_input = str(input('Would you like a fixed input or a random input pattern (y/n)?\n'
                                    'If you type "y", the following input pattern will be used:\n'
                                    ' Input pattern:    [.99, .01, .99, .01, .99, .01]\n'
                                    ' Output pattern:   [.99, .99, .01, .01]\n'
                                    'If you type "n", random input-, and output patterns will be generated\n'
                                    'What will it be? '))
            if which_input not in ['yes', 'y', 'no', 'n']:
                print("Unexpected input. Please type 'y' or 'n' only.\n")
                continue
            else:
                actual_input = []
                desired_output = []
                if which_input in ['y', 'yes']:
                    actual_input = [.99, .01, .99, .01, .99, .01]
                    desired_output = [.99, .99, .01, .01]
                else:
                    [actual_input.append(round(random.uniform(-1, 1), 2)) for i in range(6)]
                    [desired_output.append(round(random.uniform(-1, 1), 2)) for j in range(4)]
                break

        actual_input, desired_output = np.array(actual_input), np.array(desired_output)
        print('\nInput pattern:\n', actual_input, '\nOutput pattern:\n', desired_output)

        while True:
            other_input = str(input('\nWould you like to use a fixed number of error adjusting loops (y/n)?\n'
                                    'If you type "y", you can define how many cycles the weights will change\n'
                                    'If you type "n", weights will change until a user-defined minimal error is '
                                    'reached\n'
                                    'Please specify your preference: '))

            if other_input not in ['yes', 'y', 'no', 'n']:
                print("Unexpected input. Please type 'y' or 'n' only.\n")
                continue
            else:
                margin = float(input('\nWhat is the maximal error you want to tolerate?\n'
                                     'An example:\n'
                                     'The desired output can be [.99, .99, 01, 01]\n\n'
                                     'Your current weights assure that your activation ends up like this:\n'
                                     '[.79, .79, .21, .21]\n'
                                     'In other words, an error of .2\n'
                                     'We can cycle until the error becomes .05 or smaller.\n'
                                     'This means that your weight matrix will lead to an activation (+-) like this:\n'
                                     '[.95, .95, .03, .03]\n\n'
                                     'So, how much error can you tolerate? '))
                if other_input in ['y', 'yes']:
                    cycles = int(input('How many cycles are we looping? '))
                    loop_delta(actual_input, desired_output, cycles, margin)
                    break
                else:
                    loop_delta_until_found(actual_input, desired_output, margin)
                    break
        print('###\n')
        the_end = str(input('End program? (y/n): '))
        if the_end in ['yes', 'y']:
            print('Terminating program ...')
            break


np.set_printoptions(suppress=True)
main()
