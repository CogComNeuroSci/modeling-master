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

# import: general and scikit-learn specific
import numpy                 as np

from sklearn                 import datasets
from sklearn.linear_model    import Perceptron
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

# import the Iris flower dataset
iris        = datasets.load_iris()
X           = ...
y           = ...

# binarize the data: we relabel 1 to 2
   # thus, the flower is either class 0 or class 2
...

# split data in training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state = 20)

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter         = ...,
                                      tol              = 1e-3,
                                      verbose          = 0,
                                      random_state     = 20,
                                      n_iter_no_change = 5)

# fit ('train') classifier to the training data
classification_algorithm.fit(..., ...)

# predict y based on x for the test data
y_pred = classification_algorithm.predict(...)

# select wrong predictions (absolute vals) and print them
...
print("Our classification was wrong for {0} out of the {1} cases.".format(..., 
                                                                          ...))


# print accuracy using dedicated function
print('Accuracy percentage: {0:.2f}'.format(accuracy_score(..., ...) * 100))