#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:06:36 2018

@author: tom verguts
extra exercise 4
some exploration of a data set
this doesn't do much... but the question was indeed just to explore a bit
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model    import Perceptron
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score


def plot_digit(row):
# this is to make a plot of a digit
    v = np.reshape(X[row,:], (8,8))
    plt.imshow(v, plt.cm.gray_r)   
    return
    
# I found this dataset by doing datasets.[TAB]
#these are 8*8 pixels handwritten digits
digits = datasets.load_digits()

# let's now extract data from the just-created object "digits"
# dataset always consists of data and target, or X and Y
X = digits.data
Y = digits.target

# how large is it?
print([X.shape, Y.shape])

# i wanna see some data; because it's so big, it will not print all
print(X)

# so better specifically ask for two rows, then at least I can see the whole row
print(X[:2,:])

# what is the range of the data values?
print([np.amin(X), np.amax(X)])

# what is the mean feature value (note: 64 features so this gives 64 means)
print(np.mean(X, axis = 0))

# and make a picture of a row of X
row = 5
plot_digit(row)

# is it linearly separable?
# the scaler
sc = StandardScaler()
# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter = 100,
                                      verbose = 0)

# split data in training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y)
sc.fit(X_train)      
# apply scaler to all x
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train_std, Y_train)
# predict y based on x for the test data
Y_pred = classification_algorithm.predict(X_test_std)

accuracy = accuracy_score(Y_test, Y_pred)

# print accuracy using dedicated function
print('Accuracy percentage: {0:.2%}'.format(accuracy))