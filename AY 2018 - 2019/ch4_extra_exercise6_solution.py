#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
original script
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 
extra exercise 6
NOT YET DONE; THIS SCRIPT IS JUST COPY-PASTE FROM EARLIER SCRIPTS
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


all_dimensions = [[1, 2, 3]]

# import the Iris flower dataset
iris = datasets.load_iris()
y = iris.target
class_names = iris.target_names

accuracy_total = np.zeros(3)

sc = StandardScaler()

for normalization_loop in range(3): # with and without; and with sd = 5
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
        elif normalization_loop == 2:
            sc.fit(X_train)      
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            X_train_std = X_train_std*5
            X_test_std = X_test_std*5
        else:
            X_train_std = X_train
            X_test_std = X_test
    
        # define classifier (Perceptron object from scikit-learn)
        classification_algorithm = Perceptron(max_iter = 200,
                                      verbose = 0, fit_intercept = False)

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

# and then finallly the result...
print("accuracy with normalization: {:.1%}%\n".format(accuracy_total[0])      +
      "accuracy without normalization: {:.1%}%\n".format(accuracy_total[1])   +
      "accuracy with std = 5 normalization: {:.1%}%".format(accuracy_total[2]))