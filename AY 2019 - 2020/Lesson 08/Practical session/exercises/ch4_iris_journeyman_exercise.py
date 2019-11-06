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
from sklearn import datasets

#%%

# import the Iris flower dataset and extract the relevant data
iris = datasets.load_iris()

# binarize the target variable: we relabel 1 to 2
   # thus, the flower is either class 0 or class 2
...

#%%

# shuffle and split your data (same proportions as exercise 1)
...

# use scikit-learn to train your model on the input and its associated target
...

#%%

# how does your model perform on the 'unseen' observations? 
...

# print accuracy
...
