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

# import: general and scikit-learn specific
import numpy              as np

...

#%%

# define the patterns we need to associate
image_1  = np.array([.99, .01, .99, .01, .99, .01])
output_1 = 0

image_2  = np.array([.01, .99, .01, .99, .01, .99])
output_2 = 1

# make an input - and an output array
  # dimensions should be (100, 6) and (100, ) respectively
...

#%%

# shuffle input - and output array **together**
...

#%%

# split the data in a training proportion (75% )and a test proportion (25%)
X_train, y_train, X_test, y_test = ...


# use scikit-learn to train your model on the input and its associated target
...

#%%

# how does your model perform on the 'unseen' observations? 
...

# print accuracy using the built-in sklearn function accuracy_score()
...
