#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""


#%%

# -------------- #
# IMPORT MODULES #
# -------------- #

import numpy as np
import os
import pickle

from sklearn.linear_model import Perceptron
from sklearn.metrics      import accuracy_score

#%%

# load in the data
os.chdir(r'C:\Users\pieter\Downloads\dogs-vs-cats\processed')

objects = []
with (open("cats_dogs.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


dataset = objects[0]

del objects

# some background
print(dataset.DESC)

# load data
X = dataset.images
y = dataset.target

# reshape images to fit the perceptron
X = np.array(X).reshape(-1, X.shape[1] * X.shape[1])
    
#%%

training_accuracy = np.zeros(500)

for i in range(len(training_accuracy)):
        
    # shuffle arrays together
    indx = np.arange(X.shape[0])
    np.random.shuffle(indx)
    
    X_shuffled = X[indx]
    y_shuffled = y[indx]
    
    # draw n samples from the entire array
    n_samples  = np.random.randint(80*80, len(X))
    
    X_shuffled = X_shuffled[:n_samples]
    y_shuffled = y_shuffled[:n_samples]
    
    # split the data in the training proportion and the test proportion
    X_train, y_train, X_test, y_test = X_shuffled[:75,:], y_shuffled[:75], \
                                       X_shuffled[75:,:], y_shuffled[75:]
    
    del indx, X_shuffled, y_shuffled
    
    # define classifier (Perceptron object from scikit-learn)
    classification_algorithm = Perceptron()
    
    # fit ('train') classifier to the training data
    classification_algorithm.fit(X_train, y_train)
    
    # predict y based on x for the test data
    y_pred = classification_algorithm.predict(X_train)
    
    # print accuracy using a built-in sklearn function
    training_accuracy[i] = accuracy_score(y_train, y_pred) * 100

