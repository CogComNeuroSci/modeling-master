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

#%%

# import: general and scikit-learn specific
import numpy              as np

from sklearn.linear_model import Perceptron
from sklearn.metrics      import accuracy_score

#%%

# define the patterns we need to associate
image_florence   = np.array([.99, .01, .99, .01, .99, .01])
group_1          = 0

image_bfmv       = np.array([.01, .99, .01, .99, .01, .99])
group_2          = 1

# make an input - and an output array
  # dimensions should be (100, 6) and (100, ) respectively
images           = np.vstack((image_florence, image_bfmv))
songs            = np.vstack((group_1, group_2))

n                = 50
image_array      = np.repeat(images, 
                             n, 
                             axis = 0)
song_array       = np.ravel(np.repeat(songs, 
                                      n, 
                                      axis = 0))

# delete unneeded items from workspace for clear scripting
del image_florence, group_1, image_bfmv, group_2, images, songs, n

#%%

# shuffle arrays together
indx = np.arange(image_array.shape[0])
np.random.shuffle(indx)

image_array = image_array[indx]
song_array  = song_array[indx]

#%%

# split the data in the training proportion and the test proportion
X_train, y_train, X_test, y_test = image_array[:75,:], song_array[:75], \
                                   image_array[75:,:], song_array[75:]

del indx, image_array, song_array

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter         = 100,
                                      tol              = 1e-3,
                                      verbose          = 0,
                                      random_state     = 2019,
                                      n_iter_no_change = 5)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

#%%

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test)

# print accuracy using a built-in sklearn function
print('Accuracy percentage: {0:.2f}'.format(accuracy_score(y_test, y_pred) * 100))
