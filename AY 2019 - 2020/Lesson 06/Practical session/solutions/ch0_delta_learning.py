#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
Contact: Pieter.Huycke@UGent.be

- - - - - - - - - - - - 

Reading the documentation of the function:
    - print(func_name.__doc__)
Reading the function arguments, and their types:
    - func_name.__annotations__
"""


#%%

import numpy as np

#%%


def activation_function(netinput : float, 
                        mode     : str = "log") -> float:
    """
    Returns the netinput at a unit after transformation through an activation
    function.

    Parameters
    ----------
    netinput : float
        The netinput at a input. Can be calculated using the linearity 
        principle

    Returns
    -------
    float
        The netinput to a unit after transformation.
    """

    if mode == "linear":
        return netinput
    else:
        return 1 / (1 + np.exp(-netinput))



def init_weights(input_pattern  : np.ndarray, 
                 output_pattern : np.ndarray, 
                 all_zero       : bool = False, 
                 fixed_val      : bool = False, 
                 verbose        : bool = True) -> np.ndarray:
    """
    Initializes a fitting weight matrix depending on the input- and output
    patterns. Can be filled with zeros, or a float of your choice.

    Parameters
    ----------
    input_pattern : np.array
        An array representing the input to our model.
    output_pattern : np.array
        An array representing the output that a model might give.
    all_zero : bool, optional
        Should the start weight matrix consist completely of zeros or not?
        The default is False.
    fixed_val : bool, optional
        Provide a fixed value, and the weight matrix will consist solely of
        this value. The default is False.
    verbose : bool, optional
        Print some messages while processing or not. The default is True.

    Returns
    -------
    A weight matrix that is in line with the dimensions of the input- and
    output patterns.
    """

    # make sure that a 1D array is provided
    assert input_pattern.ndim == output_pattern.ndim == 1

    # get the length of the arrays for convenience
    len_in, len_out = len(input_pattern), len(output_pattern)

    # decision tree to cope with the needs of the modeller
    if fixed_val:
        val   = float(input("What value should be used?\nValue: "))
        check = str(input("Are you sure about your choice? [y / n]\n")).lower()
                      
        while check != "y":
            val   = float(input("What value should be used?\nValue: "))
            check = str(input("Are you sure about your choice? [y / n]\n")).lower()
        else:
            return np.full((len_out, len_in), val, dtype=float)
    else:
        if all_zero:
            if verbose:
                print('Using zeros to fill the array...\n')
            else:
                pass
            return np.zeros((len_out, len_in))
        else:
            if verbose:
                print('Using random floats drawn from a normal distribution to fill the array...\n')
            else:
                pass
            return np.random.randn(len_out, len_in)



def internal_input(input_pattern : np.ndarray, 
                   w_matrix      : np.ndarray):
    """
    Returns the activation at a unit, where the activation is calculated
    by inputting the summed activation into an activation function.

    Parameters
    ----------
    input_pattern : np.array
        An array representing an input to the model.
    w_matrix : np.array
        An array representing the weight matrix, can be created using 
        'init_weights( )'.

    Returns
    -------
    An array containing all the activations at output level. The array has
    N elements, where N equals the number of rows in w_matrix.
    """

    # a quick check to avoid errors
    assert w_matrix.ndim <= 2
    
    # alloacte space and compute the activation levels at each output
    activations = np.zeros(w_matrix.shape[0])
    for i in range(len(activations)):
        activations[i] = activation_function(np.dot(input_pattern, w_matrix[i]))

    return activations



def weight_change(input_pattern  : np.ndarray,
                  output_pattern : np.ndarray, 
                  w_matrix       : np.ndarray,
                  step           : float = .5,
                  keep_orig      : bool  = False) -> np.ndarray:
    """
    The actual Delta learning algorithm. Running this function once will lead
    to a single change in the weights. Run multiple times for optimal result.

    Parameters
    ----------
    input_pattern : np.ndarray
        The array representing input to your model.
    output_pattern : np.ndarray
        The array representing the output of your model.
    w_matrix : np.ndarray
        The weight matrix that will be altered when Delta learning.
    step : float, optional
        The step size used in Delta learning. A higher value leads to larger
        weight changes in a single run. The default is .5.
    keep_orig : bool, optional
        Boolean indicating whether the original weight matrix should be kept 
        or not. The default is False.

    Returns
    -------
    w_matrix : np.ndarray
        A new weight matrix that is altered due to the Delta learning.
    """

    # suppress scientific output when printing arrays (because it's ugly)
    np.set_printoptions(suppress = True)
    
    # get the activation at output with the original weight matrix
    intern_act = internal_input(input_pattern, 
                                w_matrix)
    
    # keep the original weights for future reference
    original = np.copy(w_matrix)
    
    # actual delta learning
    for i in range(len(output_pattern)):
        delta = np.array(output_pattern[i]) - intern_act[i]
        for j in range(len(w_matrix[i])):
            w_matrix[i][j] += step * input_pattern[j] * delta \
                                   * intern_act[i] * (1 - intern_act[i])

    # return the weights after learning, and possibly the original matrix too
    if keep_orig:
        original, w_matrix
    return w_matrix


