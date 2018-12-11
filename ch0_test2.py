#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:50:41 2018

@author: tom verguts
define and fit the classifiers for Test2
"""

import numpy as np
import warnings
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
file_name = "test2_data.npy"
data = np.load(file_name)

X =  data[:,[0,1]]
y = [data[:,2], data[:,3]]

threshold = 0.8 # when do we deem classification high enough?
max_hidden_size = 10 # the max number of hidden units to be tested

counter = 0
for y_labels in y:
    print("dataset nr...{}".format(counter))
    for hidden_size in range(max_hidden_size):
        if hidden_size == 0:
            classif = Perceptron()
        else:
            classif = MLPClassifier(hidden_layer_sizes = (hidden_size,), max_iter = 10000, tol = .0001)
        classif.fit(X, y_labels)
        y_pred = classif.predict(X)
        acc = accuracy_score(y_labels, y_pred)
        #print("accuracy in classification = {:.0%}".format(acc))
        if acc > threshold:
            break    
    print("total number of units needed = {}".format(3+hidden_size))
    counter += 1