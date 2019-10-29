ole#!/usr/bin/python3
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

# import modules
import ch0_delta_learning as delta_learning
import numpy              as np

# alter print options for numpy: suppress scientific printing 
np.set_printoptions(suppress = True)

image_florence   = [.99, .01, .99, .01, .99, .01]     # represents image
song_stand_by_me = [.99, .99, .01, .01]               # represents song

# define a weight matrix exclusively filled with zeros
weight_matrix = delta_learning.initialise_weights(image_florence, 
                                                  song_stand_by_me, 
                                                  zeros      = True,
                                                  predefined = False, 
                                                  verbose    = True)

# show me what you got 
print('Our original weight matrix, for now filled with zeros:\n', 
      weight_matrix)

# make a copy of the original weight matrix
original_weight_matrix = np.copy(weight_matrix)

#%%

# activation associated with the all zero weight matrix
activation_original = delta_learning.internal_input(image_florence,
                                                    weight_matrix)[0]
print('\nActivation levels at output for the original weight matrix:\n', 
      activation_original)

#%%

loops = 1000
alpha = 1.5
    
for loop_var in np.arange(1, loops + 1):
    weights_after_learning = delta_learning.weight_change(alpha,
                                                          image_florence,
                                                          song_stand_by_me,
                                                          weight_matrix)
    weight_matrix = weights_after_learning


print('\nOur altered weight matrix after {} trials of delta learning:\n'.format(loops), 
      weight_matrix)

#%%

# activation associated with this altered weight matrix
activation_after_learning = delta_learning.internal_input(image_florence,
                                                          weight_matrix)[0]
print('\nActivation levels at output after {} trials of delta learning:\n'.format(loops), 
      np.round(activation_after_learning, 3))