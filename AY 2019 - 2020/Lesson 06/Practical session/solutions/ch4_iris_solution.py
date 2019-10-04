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
from sklearn.preprocessing   import StandardScaler

# import the Iris flower dataset
iris        = datasets.load_iris()
X           = iris.data
y           = iris.target
class_names = iris.target_names

# split data in training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state = 20)

# train the standard scaler with a part of the data
sc = StandardScaler()
sc.fit(X_train)

# apply scaler to all x
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter         = 100,
                                      tol              = 1e-3,
                                      verbose          = 0,
                                      random_state     = 20,
                                      n_iter_no_change = 5)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train_std, y_train)

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test_std)

# select wrong predictions (absolute vals) and print them
compared       = np.array(y_pred == y_test)
absolute_wrong = (compared == False).sum()
print("Our classification was wrong for {0} out of the {1} cases.".format(absolute_wrong, 
                                                                          len(compared)))


# print accuracy using dedicated function
print('Accuracy percentage: {0:.2f}'.format(accuracy_score(y_test, y_pred) * 100))