#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""


#%%

# import: general and scikit-learn specific
import numpy             as np
import os

from sklearn.linear_model    import Perceptron
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

#%%

# load the data
location = r'C:\Users\pieter\Downloads\GitHub\modeling-master\AY 2019 - 2020\Lesson 10\Practical session\exercises'
data     = np.load(os.path.join(location, 'ch5_circular_dataset_novice.npy'))

X = data[:,:2]
y = data[:,-1]

#%%

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
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
classification_algorithm = MLPClassifier(hidden_layer_sizes = (1000, ),
                                         max_iter           = 20000)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test)

# print accuracy using a built-in sklearn function
print('MLP accuracy:\n\t {0:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))