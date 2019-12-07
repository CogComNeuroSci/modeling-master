#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 

Stuck using a function?
No idea about the arguments that should be defined?

Type:
help(module_name)
help(function_name)
to let Python help you!
"""


#%%

# -------------- #
# IMPORT MODULES #
# -------------- #

import ch0_delta_learning as delta_learning
import numpy              as np

# alter print options for numpy: suppress scientific printing 
np.set_printoptions(suppress=True)

#%%

# ----------------------------------------- #
# LEARN ASSOCIATION BETWEEN TONE AND SHOCK #
# ----------------------------------------- #

# define inputs
tone           = np.array([1, 0])
tone_and_light = np.array([1, 1])
shock          = np.array([1])

# define a weight matrix exclusively filled with zeros
weight_matrix = delta_learning.initialise_weights(tone, 
                                                  shock, 
                                                  zeros      = True,
                                                  predefined = False, 
                                                  verbose    = False)
   
# actual learning
loops = 1000
alpha = 1.5
    
for loop_var in np.arange(1, loops + 1):
    weights_after_learning = delta_learning.weight_change(alpha,
                                                          tone,
                                                          shock,
                                                          weight_matrix)
    weight_matrix = weights_after_learning
    
# show that the light leads to the desired response
activation_after_learning = delta_learning.internal_input(tone,
                                                          weight_matrix)[0]
print('\nActivation of output unit after {} trials of delta learning:\n'.format(loops), 
      np.round(activation_after_learning, 3))

#%%

# make a copy of the original weight matrix
weight_matrix_light = np.copy(weight_matrix)

# ------------------------------------------------ #
# LEARN ASSOCIATION BETWEEN TONE + LIGHT AND SHOCK #
# ------------------------------------------------ #
  
# actual learning
loops = 1000
alpha = 1.5
    
for loop_var in np.arange(1, loops + 1):
    weights_after_learning = delta_learning.weight_change(alpha,
                                                          tone_and_light,
                                                          shock,
                                                          weight_matrix)
    weight_matrix = weights_after_learning
    
# show that the light + tone leads to the desired response
activation_after_learning = delta_learning.internal_input(tone_and_light,
                                                          weight_matrix)[0]
print('\nActivation levels at output after {} trials of delta learning:\n'.format(loops), 
      np.round(activation_after_learning, 3))

#%%

# --------------------- #
# THE PROOF OF BLOCKING #
# --------------------- #

# -------- #
# OPTION 1 #
# -------- #

light = np.array([0, 1])

# show that the light + tone leads to the desired response
activation_after_learning = delta_learning.internal_input(light,
                                                          weight_matrix)[0]
print('\nActivation levels at output after {} trials of delta learning:\n'.format(loops), 
      np.round(activation_after_learning, 3))

'''
Result: after training, presenting the tone alone will not lead to an active
        output unit.
        thus, the subject will not be scared from the tone alone
'''

# -------- #
# OPTION 2 #
# -------- #

if np.array_equal(weight_matrix_light[0], weight_matrix[0]):
    print('No association between the "tone" unit and the output unit was learned')

'''
Result: the weight matrix did not change after the first learning period
        thus, no association was learned between the tone unit and the output
'''