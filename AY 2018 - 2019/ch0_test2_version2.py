#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:14:34 2019

@author: tom
"""

import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:,[1, 2]]
y = iris.target

nreps = 15
learning_rate_list = [0.0001, 0.1, 0.3, 0.6, 1, 2]
accuracy = np.zeros(len(learning_rate_list))

for learning_rate_loop in range(len(learning_rate_list)):
    # split data in training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mlp = MLPClassifier(hidden_layer_sizes=(2,), 
                    max_iter=1000,
                    verbose=0, 
                    learning_rate_init=learning_rate_list[learning_rate_loop], 
                    activation='logistic')
    for rep_loop in range(nreps):
        # fit  classifier to the training data
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        accuracy[learning_rate_loop] += accuracy_score(y_test, y_pred)
    accuracy[learning_rate_loop] /= nreps
    print("accuracy with learning rate {0} equals {1:.2f}%".format(learning_rate_list[learning_rate_loop], accuracy[learning_rate_loop]))