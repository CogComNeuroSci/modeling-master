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

    number_of_weights = len(input_pattern) * len(output_pattern)

    weights_suboptimal = []
    weights = []
    [weights_suboptimal.append(round(random.uniform(-5, 5), 2)) for i in range(number_of_weights)]
    [weights.append(weights_suboptimal[j:j + len(input_pattern)]) for j in range(0, len(weights_suboptimal), len(input_pattern))]

    return weights


def internal_input(input_pattern, output_pattern, weights):

    activations = []

    for i in range(len(output_pattern)):
        added_activation = np.sum(np.multiply(input_pattern, weights[i]))
        rounded_activation = np.round(added_activation, 2)
        result = logistic_activation(rounded_activation)
        activations.append(result)

    return activations


def weight_change(alpha, input_pattern, output_pattern, weight_matr, internal_activations):

    for l in range(len(output_pattern)):
        delta = output_pattern[l] - internal_activations[l]
        for m in range(len(input_pattern)):
            weight_matr[l][m] = round(weight_matr[l][m] + (alpha*logistic_activation(weight_matr[l][m]) * delta *
                                                           output_pattern[l] * (1 - output_pattern[l])), 2)

    return weight_matr


def loop_delta(input_pattern, output_pattern, loops, tolerated_error):

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
        print('\n-- Learning completed without satisfying the defined conditions --'
              '\nActivation computed after last cycle: ', new_internal_activation,
              '\nAccompanying weight matrix: ', weight_matrix)
    return weight_matrix


def loop_delta_until_found(input_pattern, output_pattern, tolerated_error):

    cycles = 0
    max_cycles = 250000
    configuration_found = False

    weight_matrix = initialise_weights(input_pattern, output_pattern)
    print('Original (random) weight matrix: \n', weight_matrix)

    int_input = internal_input(input_pattern, output_pattern, weight_matrix)
    alpha = float(input('\nDefine a constant which influences how large the weight change each trial will be: '))
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
                  '\nTerminating search...')

    return weight_matrix


def main():

    while True:
        while True:
            which_input = str(input('Would you like a fixed input or a random input (y/n)?\n'
                                    'If you type "y", the following input will be used:\n'
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
                                    'If you type "n", weights will change until a user-defined minimal error is reached\n'
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
                else:
                    loop_delta_until_found(actual_input, desired_output, margin)
                break
        print(' --- \n')
        the_end = str(input('End program? (y/n): '))
        if the_end in ['yes', 'y']:
            print('Terminating program ...')
            break


main()
