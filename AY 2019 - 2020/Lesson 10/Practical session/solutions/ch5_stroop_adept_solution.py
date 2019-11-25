#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""


#%%

# import: general and scikit-learn specific
import numpy as np

from sklearn.linear_model    import Perceptron
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

#%%

'''
* coding * 
    - Unit 1: context (1 = word is relevant dimension)
    - Unit 2: color   (1 = word is colored in red)
    - Unit 3: word    (1 = RED is shown on the screen)
Mind that the coding does not impact your model's performance as long as one
is using a consistent coding scheme
'''

# define the input patterns
in_1 = np.array([1, 1, 1]) 
in_2 = np.array([1, 0, 0]) 
in_3 = np.array([1, 1, 0])
in_4 = np.array([1, 0, 1])
in_5 = np.array([0, 1, 1])
in_6 = np.array([0, 0, 0])
in_7 = np.array([0, 1, 0])
in_8 = np.array([0, 0, 1])

# define the targets
t1 = np.array( [1])
t2 = np.array([-1])
t3 = np.array( [1])
t4 = np.array([-1])

t5 = np.array( [1])
t6 = np.array([-1])
t7 = np.array([-1])
t8 = np.array( [1])

#%%

# zip them together
trained_input  = np.vstack((in_1, in_2, in_3, in_4))
trained_input  = np.tile(trained_input, (3,1))

trained_output = np.vstack((t1,   t2,   t3,   t4))
trained_output = np.tile(trained_output, (3,1))

input_arr  = np.vstack((trained_input,  in_5, in_6, in_7, in_8))
target_arr = np.vstack((trained_output, t5,   t6,   t7,   t8))

del in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8
del t1,   t2,   t3,   t4,   t5,   t6,   t7,   t8
del trained_input, trained_output

inputs  = np.tile(input_arr, (50,1))
targets = np.tile(target_arr, (50,1))
targets = np.ravel(targets)

del input_arr, target_arr

#%%

# train test split
X_train, X_test, y_train, y_test = train_test_split(inputs, 
                                                    targets,
                                                    train_size = .25)

#%%

# ---------- #
# PERCEPTRON #
# ---------- #

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter         = 10000,
                                      tol              = 1e-3,
                                      verbose          = 0,
                                      n_iter_no_change = 10)

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
classification_algorithm = MLPClassifier(hidden_layer_sizes = (8, ),
                                         max_iter           = 10000)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

# learning disability (actively altering the weights of the model)
for indx in range(len(classification_algorithm.coefs_)):
    shape     = classification_algorithm.coefs_[indx].shape
    mu, sigma = 0, 5
    noise     = np.random.normal(mu, sigma, (shape[0], shape[1]))
    classification_algorithm.coefs_[indx] += noise 

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test)

# print accuracy using a built-in sklearn function
print('MLP accuracy:\n\t {0:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))
