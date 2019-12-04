#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""


#%%

# import: general and scikit-learn specific
import numpy as np

from sklearn.linear_model    import Perceptron
from sklearn.model_selection import train_test_split

#%%

'''
* coding * 
    - Unit 1: target direction  ([0, 1] = >)
    - Unit 2: prime direction   ([0, 1] = >>>>)
Mind that the coding does not impact your model's performance as long as one
is using a consistent coding scheme
'''

# define the input patterns
input_arr = np.array([[0, 1, 0, 1],      # target: > / prime: >>>>
                      [1, 0, 1, 0],      # target: < / prime: <<<<
                      [0, 1, 1, 0],      # target: > / prime: <<<<
                      [1, 0, 0, 1]])     # target: < / prime: >>>>

# define the associated outputs
output_arr = np.array([ [1],             # response: RIGHT
                       [-1],             # response: LEFT
                        [1],             # response: RIGHT
                       [-1]])            # response: LEFT

# repeat the data a bit and fix the dimensions
X = np.repeat(input_arr,  
              45, 
              axis = 0)
y = np.repeat(output_arr, 
              45, 
              axis = 0)
y = y.reshape((len(y)))

del input_arr, output_arr

#%%

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    train_size = .75)

#%%

# ---------- #
# PERCEPTRON #
# ---------- #

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter         = 10,
                                      tol              = 1e-3,
                                      verbose          = 0)


# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

# learning disability (actively altering the weights of the model)
mu, sigma = 0, 5
noise     = np.random.normal(mu, sigma, (2, ))
classification_algorithm.coef_[0][-2:] += noise 

#%%

# lists to store the accuracy, no arrs because the length is unknown
acc_congr   = []
acc_incongr = []

# loop over all test cases, and store the accuracy depending on (in)congruency
for i in range(len(X_test)):
    
    y_pred  = classification_algorithm.predict([X_test[i]])
    
    if np.all(np.equal(X_test[i][:2], X_test[i][2:])):   # congruent
        if y_pred == y_test[i]:
            acc_congr.append(1)
        else:
            acc_congr.append(0)
    else:                                                # incongruent
        if y_pred == y_test[i]:
            acc_incongr.append(1)
        else:
            acc_incongr.append(0)

# print the end results
print('Accuracy for the congruent trials: {0:.2f}%'.format(np.mean(acc_congr) * 100))
print('Accuracy for the incongruent trials: {0:.2f}%'.format(np.mean(acc_incongr)* 100))