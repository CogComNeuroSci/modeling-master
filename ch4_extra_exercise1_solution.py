#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
original script
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 
extra exercise 1
adapted by tom verguts; i wanted to check which dimensions in the iris 
data allow linear separability
all dimensions are systematically combined; the code scales to an arbitary
number of dimensions
of course, some threshold has to be set on when we consider accuracy sufficiently
good to say that a dataset is linearly separable
"""

# import: general and scikit-learn specific

from sklearn                 import datasets
from sklearn.linear_model    import Perceptron
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

# below this threshold, we classify the data as "not linearly separable"
threshold = 0.9 

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

sc = StandardScaler()

for dimension_loop in all_dimensions:
    #create a dataset with a limited number of input dimensions
    X = iris.data[:,dimension_loop]

    # split data in training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=2018)

    # train the standard scaler with a part of the data
    sc.fit(X_train)

    # apply scaler to all x
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # define classifier (Perceptron object from scikit-learn)
    classification_algorithm = Perceptron(max_iter = 200,
                                      verbose = 0,
                                      random_state = 2018)

    # fit ('train') classifier to the training data
    classification_algorithm.fit(X_train_std, y_train)

    # predict y based on x for the test data
    y_pred = classification_algorithm.predict(X_test_std)

    # print accuracy using dedicated function
    print("\ncheck dimension(s) " + str(dimension_loop) + "...")

    accuracy = accuracy_score(y_test, y_pred)

    print('Accuracy percentage: {0:.2%}'.format(accuracy))

    # check linear separability
    lin_string = ["not", ""][(accuracy>threshold)*1]

    # print linear separability
    print('We consider this data ' + lin_string + ' linearly separable')