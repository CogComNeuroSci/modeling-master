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
image_1  = np.array([.99, .01, .99, .01, .99, .01])
output_1 = 0

image_2  = np.array([.01, .99, .01, .99, .01, .99])
output_2 = 1

# make an input - and an output array
  # dimensions should be (100, 6) and (100, ) respectively
images           = np.vstack((image_1, image_2))
output           = np.vstack((output_1, output_2))

n                = 50
image_array      = np.repeat(images, 
                             n, 
                             axis = 0)
output_array     = np.ravel(np.repeat(output, 
                                      n, 
                                      axis = 0))

# delete unneeded items from workspace for clear scripting
del image_1, output_1, image_2, output_2, images, output, n

#%%

# shuffle arrays together
indx = np.arange(image_array.shape[0])
np.random.shuffle(indx)

image_array  = image_array[indx]
output_array = output_array[indx]

#%%

# split the data in the training proportion and the test proportion
X_train, y_train, X_test, y_test = image_array[:75,:], output_array[:75], \
                                   image_array[75:,:], output_array[75:]

del indx, image_array, output_array

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
