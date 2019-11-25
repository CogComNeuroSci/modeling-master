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

# import the most relevant modules
import matplotlib.pyplot as plt
import numpy             as np

from   sklearn                 import datasets
from   sklearn.linear_model    import Perceptron
from   sklearn.metrics         import accuracy_score

#%%

# load the data
digits = datasets.load_digits()

X      = digits.images
y      = digits.target

# flatten the array to use it in the classifier
n_samples = len(X)
data      = X.reshape((n_samples, -1))

#%%

# shuffle arrays together
indx = np.arange(data.shape[0])
np.random.shuffle(indx)

data = data[indx]
y    = y[indx]

# binarize: 7 vs not 7
y[np.where(y != 7)] = 0
y[np.where(y == 7)] = 1


#%%

# split the data in the training proportion and the test proportion
percent_75 = round(len(X) * .75)
X_train, y_train, X_test, y_test = data[:percent_75,:], y[:percent_75], \
                                   data[percent_75:,:], y[percent_75:]
# classifier
classification_algorithm = Perceptron(max_iter         = 100,
                                      tol              = 1e-3,
                                      verbose          = 0,
                                      n_iter_no_change = 5)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

# predict y based on x for the test data
y_pred = classification_algorithm.predict(X_test)

# accuracy
print('Accuracy percentage: {0:.2f} %'.format(accuracy_score(y_test, y_pred) * 100))

#%%

mapping_preds = {0: 'Not 7', 
                 1: '7'}

# show some input images and their predicted label
images_and_predictions = list(zip(X[indx][percent_75:,:], y_pred))

# get some random index to print (show a predicted 7 and some "Not 7"'s)
random_indx = np.random.choice(np.where(y_pred == 1)[0])

# print some predictions
for index, (image, prediction) in enumerate(images_and_predictions[random_indx:random_indx+4]):
    plt.subplot(1, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, 
               cmap=plt.cm.gray_r, 
               interpolation='nearest')
    plt.title('Prediction: %s' % mapping_preds.get(prediction))

plt.show()