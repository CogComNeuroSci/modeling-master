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

import numpy             as np
import os
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import accuracy_score

#%%

# load in the data
os.chdir(r'C:\Users\pieter\Downloads\GitHub\modeling-master\AY 2019 - 2020\Lesson 10\Practical session\solutions\downloads\processed')

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
    
# shuffle arrays together
indx = np.arange(X.shape[0])
np.random.shuffle(indx)

X_shuffled = X[indx]
y_shuffled = y[indx]


#%%

# split the data in the training proportion and the test proportion
X_train, y_train, X_test, y_test = X_shuffled[:29,:], y_shuffled[:29], \
                                   X_shuffled[29:,:], y_shuffled[29:]

del indx, X_shuffled, y_shuffled

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = MLPClassifier(hidden_layer_sizes = (100, 50, 8,),
                                         activation         = 'logistic',
                                         solver             = 'sgd',
                                         learning_rate      = 'constant',
                                         max_iter           = 2000, 
                                         early_stopping     = True)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test)

#%%

# print accuracy using a built-in sklearn function
print('Accuracy percentage: {0:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))

