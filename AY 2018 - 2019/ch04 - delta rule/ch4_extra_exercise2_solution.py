#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 
extra exercise 2
adapted by tom verguts; based on code of exercise 1; I will now systematically 
put on or off the normalization
"""

# import: general and scikit-learn specific
import numpy as np

from sklearn                 import datasets
from sklearn.linear_model    import Perceptron
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

# below this threshold, we classify the data as "not linearly separable"
threshold = 0.9 

# do we want lines and lines of intermediate results?
verbose = False

all_dimensions = []
n_dim = 4

# these are the ones I will check
for loop in range(2**n_dim - 1):
    binary_number = bin(loop+1)[2:]
    binary_number = "0"*(n_dim-len(binary_number)) + binary_number
    dimension_vector = []
    for dim_loop in range(n_dim):
        if int(binary_number[dim_loop]) == 1:
            dimension_vector.append(dim_loop)
    all_dimensions.append(dimension_vector)
    
# import the Iris flower dataset
iris = datasets.load_iris()
y = iris.target
class_names = iris.target_names

accuracy_total = np.zeros(2)

# the scaler
sc = StandardScaler()
# define classifier (Perceptron object from scikit-learn)
# I took away the random starting state; therefore, each run will generate
# slightly differen results
classification_algorithm = Perceptron(max_iter = 200,
                                      verbose = 0)

for normalization_loop in range(2): # with and without
    for dimension_loop in all_dimensions:
        #create a dataset with a limited number of input dimensions
        X = iris.data[:,dimension_loop]
        
        # split data in training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y)

        # train the standard scaler with a part of the data
        if normalization_loop == 0:
            sc.fit(X_train)      
            # apply scaler to all x
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
        else:
            X_train_std = X_train
            X_test_std = X_test

        # fit ('train') classifier to the training data
        classification_algorithm.fit(X_train_std, y_train)

        # predict y based on x for the test data
        y_pred = classification_algorithm.predict(X_test_std)

        if verbose:
            print("\ncheck dimension(s) " + str(dimension_loop) + "...")

        accuracy = accuracy_score(y_test, y_pred)

        # print accuracy using dedicated function
        if verbose:
            print('Accuracy percentage: {0:.2%}'.format(accuracy))

        # check linear separability
        lin_string = ["not", ""][(accuracy>threshold)*1]

        # print linear separability
        if verbose:
            print('We consider this data ' + lin_string + ' linearly separable')
        
        accuracy_total[normalization_loop] += accuracy

# this works because it's an nparray...
accuracy_total = accuracy_total / (2**n_dim - 1)

# and then finallly the result...
print("accuracy with normalization: {0:.1%}%;\naccuracy without normalization: {1:.1%}%".format(accuracy_total[0], accuracy_total[1]))