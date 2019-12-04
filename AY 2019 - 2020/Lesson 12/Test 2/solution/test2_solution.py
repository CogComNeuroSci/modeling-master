#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""

#%%

# import modules
import itertools
import numpy as np

from sklearn.linear_model    import Perceptron
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

#%%

# make all combinations and assign to an array or list (situation different)
combinations     = list(itertools.product([0, 1, 2, 3], repeat = 3))
separate_strings = []
grammar_coding   = np.zeros(len(combinations))

neural_coding    = {0 : [0, 0, 0, 1],  # A, coded as 0, with pattern [0,0,0,1]
                    1 : [0, 0, 1, 0],  # B, coded as 1, with pattern [0,0,1,0]
                    2 : [0, 1, 0, 0],  # C, coded as 2, with pattern [0,1,0,0]
                    3 : [1, 0, 0, 0]}  # D, coded as 3, with pattern [1,0,0,0]

# loop over all combinations; join strings and add coding
for indx in range(len(combinations)):
    
    # a temporary array to map '0' (A) to [0, 0, 0, 1]
    temp = []
    for arrs in combinations[indx]:
        temp.append(neural_coding.get(arrs))
    # glue the final code and save
    separate_strings.append(np.ravel(temp))
    
    # coding depending on grammar or not
    if temp[0] == temp[-1]:
        grammar_coding[indx] = 1
    else:
        grammar_coding[indx] = 0

# make separate_strings an array
separate_strings = np.array(separate_strings)

del combinations, indx, arrs, temp, neural_coding

#%%
        
# make an elaborate training -, and test set
X = np.repeat(separate_strings, 50, axis = 0)
y = np.repeat(grammar_coding, 50)

#%%

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    train_size = .75)

del grammar_coding, separate_strings

#%%

# ---------- #
# PERCEPTRON #
# ---------- #

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter         = 10000,
                                      tol              = 1e-3,
                                      verbose          = 0)


# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test)

# print accuracy using a built-in sklearn function
print('Perceptron accuracy:\n\t {0:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))

#%%

# --- #
# MLP #
# --- #

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = MLPClassifier(hidden_layer_sizes = (3, ),
                                         max_iter           = 20000)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test)

# print accuracy using a built-in sklearn function
print('MLP accuracy:\n\t {0:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))