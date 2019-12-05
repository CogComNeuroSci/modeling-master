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

SIMULATIONS = 50
REPETITIONS = 50
MAX_HIDDEN  = 20

#%%

# make all combinations and assign to an array or list (situation different)
combinations     = list(itertools.product([0, 1], repeat = 3))
separate_strings = []
grammar_coding   = np.zeros(len(combinations))

neural_coding    = {0 : [0, 1],  # A, coded as 0, with pattern [0,1]
                    1 : [1, 0]}  # B, coded as 1, with pattern [1,0]


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
X = np.repeat(separate_strings, REPETITIONS, axis = 0)
y = np.repeat(grammar_coding, REPETITIONS)

del grammar_coding, separate_strings

#%%

print('\n- - - -\nPerceptron\n- - - -\n')

#%%

# ---------- #
# PERCEPTRON #
# ---------- #

perceptron_acc = np.zeros(SIMULATIONS)

for i in range(SIMULATIONS):
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        train_size = .75)

    # define classifier (Perceptron object from scikit-learn)
    classification_algorithm = Perceptron(max_iter         = 10000,
                                          tol              = 1e-3,
                                          verbose          = 0)
    
    
    # fit ('train') classifier to the training data
    classification_algorithm.fit(X_train, y_train)
    
    # predict y based on x for the test data
    y_pred = classification_algorithm.predict(X_test)

    perceptron_acc[i] = accuracy_score(y_test, y_pred) * 100

print('Average accuracy of our Perceptron: {0:.2f}%\n'.format(np.mean(perceptron_acc)))

#%%

print('- - - -\nMulti-layered Perceptron\n- - - -\n')

#%%

# --- #
# MLP #
# --- #

for hidden_units in range(1, MAX_HIDDEN + 1):
    
    unsatisfied = False
    
    for loop_number in range(SIMULATIONS):
        
        # define classifier (Perceptron object from scikit-learn)
        classification_algorithm = MLPClassifier(hidden_layer_sizes = (hidden_units, ),
                                                 max_iter           = 10000, 
                                                 n_iter_no_change   = 10)
        
        # fit ('train') classifier to the training data
        classification_algorithm.fit(X_train, y_train)
        
        # predict y based on x for the test data
        y_pred = classification_algorithm.predict(X_test)
        
        # print accuracy using a built-in sklearn function
        if int(accuracy_score(y_test, y_pred)) < 1.0:
            unsatisfied = True
            break
        
    if unsatisfied:
        if hidden_units == 1:
            print('{} hidden unit: unsatisfactory'.format(hidden_units))
        else:
            print('{} hidden units: unsatisfactory'.format(hidden_units))
    else:
        print('We have 100% accuracy when using {} hidden units'.format(hidden_units))
        break
