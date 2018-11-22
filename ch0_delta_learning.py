#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
Contact: Pieter.Huycke@UGent.be

- - - - - - - - - - - - 

This 'module' was made with the intention to create a collection of functions
that are capable of performing 'Delta learning' (as defined in the Modelling of 
Cognitive Processes course).

These functions can be used for basic computations where Delta learning is 
appropriate. For large datasets, we recommend the use of the dedicated 
machine learning module 'scikit-learn'.
"""


import numpy as np


def activation_function(netinput, form='logistic'):

    """
    :param netinput:
    The netinput for a certain input unit
    This is calculated using the linearity principle (see chapter 4, page 1)

    :param form:
    Determines the type of the activation function that is used
    This parameter can have two values:
        'linear' means that a linear activation function is used to 
        transform the netinput
        'logistic' means that a logistic activation function is used to 
        transform the netinput
    String: default value set to "logistic"

    :return:
    This function returns the netinput after first transforming the netinput
    using an activation function.
    Two types of transformations are possible: linear or logistic.
    
    :example use:
    activation_function(2.7)                  # 'logistic' form is assumed
    activation_function(2.7, form='linear')   # 'linear' defined by user
    """

    if form == 'linear':
        return netinput
    else:
        return 1 / (1 + np.exp(-netinput))


def initialise_weights(input_pattern, output_pattern, zeros=False, 
                       predefined=False, verbose=False):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99, .01, .99, .01, .99, .01]
    What this input pattern represents is specified by the user
    Input patterns should be lists

    :param output_pattern:
    The output pattern which is provided
    An example: [.99 .99 .01 .01]
    What this output pattern represents is specified by the user
    Output patterns should be lists

    :param zeros:
    Defines how you want to create the initial weight matrix
    This parameter can have two values:
        True means that all weights will be set to zero
        False means that all weights will be determined randomly 
        by sampling random floats from a univariate "normal" (Gaussian) 
        distribution of mean 0 and variance 1
    Boolean: default value set to False

    :param predefined:
    This parameter defines whether the weight matrix is predefined or not
    The default value is 'False'
    When the value is True, this means that the weight matrix 
    (in the correct form) should be provided by the user.
    An option is provided to make a weight matrix of the correct format 
    that contains 1 predefined float value.
    Boolean: default value set to False

    :return:
    This function returns a weight matrix
    The weight matrix is a numpy array will have the following dimensions:
        weight_matrix.shape = (n, m)
    This means that the weight matrix is an array that has n 'subarrays'
    Each 'subarray' contains m elements, note that:
    n = len(output pattern)
    m = len(input pattern)
    This weight matrix signifies the weights from all m input units to the 
    n output units
    Consequently, the first 'subarray' represents the weights from all 
    m input units to the first of the n output units.
    
    :example use:
    initialise_weights(sight_dog, happiness)
    
    where 'sight_dog' and 'happiness' are numpy arrays representing these
    concepts
    """

    if predefined:
        answers = str(input('\nPlease input your predefined weight matrix\n'
                            'Note that the weight matrix should comply to a specific form:\n'
                            'Example: if you have 4 input units, and 2 output units, then you should have a list with '
                            '8 weights in total.\n'
                            'This because 4 * 2 = 8.\n'
                            'You should have a large list with two (the number of output units) sublists in it.\n'
                            'Each sublist should contain 4 weights (the number of input units)\n'
                            'A valid example would be:\n'
                            '[[.5, .2, .9, .1], [.3, .3, 6, .7]]\n\n'
                            'Make sure your format is correct...\n'
                            'Would you like a weight matrix with 1 value, in the correct format?\n'
                            'If so, you can press "y".\n'
                            'Otherwise, you can copy-paste your weight matrix in the prompt.\n'
                            'Answer (y/n): '))
        if answers.lower() in ['yes', 'y']:
            predefined_val = float(input('The value in your weight matrix? '))
            return np.full((len(output_pattern), len(input_pattern)), predefined_val)
        else:
            weights = input('The weight matrix: ')
            return weights
    else:
        if zeros:
            if verbose:
                print('Using zeros to fill the array...\n')
            else:
                pass
            return np.zeros(shape=(len(output_pattern), len(input_pattern)))
        else:
            if verbose:
                print('Using random floats drawn from a normal distribution to fill the array...\n')
            else:
                pass
            return np.random.randn(len(output_pattern), len(input_pattern))


def internal_input(input_pattern, weights, act_function='logistic'):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    What this input pattern represents is specified by the user
    Input patterns should be lists

    :param weights:
    The weight matrix representing the weights between our input-, and 
    output pattern.
    This matrix can be initialised using the function 'initialise_weights()'.

    :param act_function:
    Determines the type of the activation function that is used
    This parameter can have two values:
        'linear' means that a linear activation function is used to
        transform the netinput
        'logistic' means that a logistic activation function is used to 
        transform the netinput
    String: default value set to "logistic"

    :return:
    Returns the netinput after it was transformed using a logistic activation 
    function
    The netinput is calculated using the linearity principle 
    (input * weights for all sending units)
    Subsequently, this summed input is transformed
    This function returns a list with all activations for all output units
    
    :example use:
    internal_input(sight_dog, happiness)
    
    where 'sight_dog' and 'happiness' are lists representing these
    concepts
    Mind that the "logistic" activation function will be used here
    """

    activations = []

    for i in range(len(weights)):
        added_activation = np.sum(np.multiply(input_pattern, weights[i]))
        activations.append(activation_function(added_activation, form=act_function))

    return activations, act_function


def weight_change(alpha, input_pattern, output_pattern, weights, 
                  function_word='logistic'):

    """
    :param alpha:
    The stepsize
    The larger this parameter is, the more drastic the weight changes in 
    each trial will be

    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    What this input pattern represents is specified by the user
    Input patterns should be lists

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    What this output pattern represents is specified by the user
    Output patterns should be lists

    :param weights:
    The weight matrix representing the weights between our input-, and 
    output pattern.
    This matrix can be initialised using the function 'initialise_weights()'.

    :param function_word:
    Determines the type of the activation function that is used
    This parameter can have two values:
        'linear' means that a linear activation function is used to 
        transform the netinput
        'logistic' means that a logistic activation function is used to 
        transform the netinput
    String: default value set to "logistic"

    :return:
    Determines the weight change for each trial based on the internal input
    The difference the desired activation level and the actual activation 
    level is used to do so
    
    :example use:
    weight_change(1.5, sight_dog, sight_dog, yielded_weight_matrix, 
                  function_word='logistic')
    
    where the stepsize is set at 1.5, and 'sight_dog' and 'happiness' are 
    (as usual) lists representing these concepts. 
    'yielded_weight_matrix' represents a numpy array representing the weights
    between the input- and the output pattern.
    This weight matrix can be predefined, or computed by this module.
    """

    np.set_printoptions(suppress=True)
    weights = np.array(weights)

    for i in range(len(output_pattern)):
        altered_weights = weights[i]
        for j in range(len(altered_weights)):
            internal_activation = internal_input(input_pattern, weights, act_function=function_word)[0]
            delta = np.array(output_pattern[i]) - internal_activation[i]
            altered_weights[j] = altered_weights[j] + \
                alpha * input_pattern[j] * delta * internal_activation[i] * (1 - internal_activation[i])
        weights[i] = np.round(altered_weights, 3)
    return weights


def asking_questions():

    """
    :return:
    A helper function to get some variables that are needed to perform Delta 
    learning
    Specifically, we want to get a value for the following parameters:
        - The initial weight matrix:
            - floats?
            - all zero?
        - The activation function:
            - linear?
            - logistic?
        - The stepsize alpha:
            - the value (treated as float)?
    All values should be defined by the user, and are stored for later 
    computation.
    Returns an answer for these three questions
    
    :example use:
    asking_questions()    # function auto-executes
    """

    while True:
        answer_one = str(input('Weights all set to zero? Type: zero\n'
                               'Weights all non-zero? Type: numbers\n'
                               'Answer: '))
        if answer_one.lower() in ['zero', 'numbers']:
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


def loop_delta(input_pattern, output_pattern, loops=50, print_loops=True, 
               error_margin=0.05):

    """
    :param input_pattern:
    The input pattern which is provided
    An example: [.99 .01 .99 .01 .99 .01]
    What this input pattern represents is specified by the user
    Input patterns should be lists

    :param output_pattern:
    The input pattern which is provided
    An example: [.99 .99 .01 .01]
    What this output pattern represents is specified by the user
    Output patterns should be lists

    :param loops:
    The number cycles where the weight matrix is altered to close the gap 
    between the actual output and the desired output
    Int: default value set to 50

    :param print_loops:
    Defines whether the alterations in the weight matrices are printed or not
    Two values are possible:
        True makes sure that all the changes in the weight matrix are printed
        False means that the above doesn't happen
    Boolean: default value is set to True

    :param error_margin:
    The allowed deviation for the activation levels.
    The outputted weight matrix will be accepted if it yields activation levels 
    that equals the output pattern +/- the error margin for each element.
    Float: default value set to 0.05

    :return:
    Changes the weight matrix for a fixed amount of cycles, or until the 
    condition is met (minimal error reached)
    
    :example use:
    loop_delta(sight_dog, happiness, loops=1000, print_loops=True, 
               error_margin=0.05)
    
    where 'sight_dog' and 'happiness' are lists representing these
    concepts. 
    We loop 1000 time, performing Delta learning with each loop.
    The changes in the weight matrix are printed for each loop.
    Our learning will be successful when the difference between the actual
    output and the desired output is smaller than 0.05.
    """

    random_zero, linear_logistic, alpha = asking_questions()

    if random_zero == 'zero':
        weights = initialise_weights(input_pattern, output_pattern, zeros=True, predefined=False)
    else:
        asking = str(input('Predefined weights?\n'
                           'Answer (y/n): '))
        if asking.lower() in ['y', 'yes']:
            weights = initialise_weights(input_pattern, output_pattern, zeros=False, predefined=True)
        else:
            weights = initialise_weights(input_pattern, output_pattern, zeros=False)

    original_weights = np.copy(weights)

    # looping
    for i in range(loops):
        new_weights = weight_change(alpha, input_pattern, output_pattern, weights,
                                    function_word=linear_logistic)
        if print_loops:
            print('New weight matrix:')
            [print(weight_matrices) for weight_matrices in new_weights]
        else:
            pass
        # testing new weight matrix
        activations = internal_input(input_pattern, new_weights, act_function=linear_logistic)[0]
        counted = 0
        for j in range(len(activations)):
            if output_pattern[j] - error_margin < activations[j] < output_pattern[j] + error_margin:
                counted += 1
            else:
                break
        if counted == len(activations):
            print('\nSatisfying weight matrix found after cycle number %d!' % (i + 1))
            break
        else:
            pass
        weights = new_weights

    outputs_original = []
    for i in range(len(original_weights)):
        outputs_original.append(activation_function(np.sum(np.multiply(original_weights[i], input_pattern))))

    print('\nOriginal weight matrix:\n', original_weights)
    print('\nOutputs yielded with original matrix:\n', outputs_original)

    outputs_new = []
    for i in range(len(weights)):
        outputs_new.append(activation_function(np.sum(np.multiply(weights[i], input_pattern))))

    print('\nAltered weight matrix:\n', weights)
    print('\nOutputs yielded with altered matrix:\n', outputs_new)
    print('\nDesired output:\n', output_pattern)

    return weights
