#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 
adapted by tom verguts for extra exercise 5
I now use SGDClassifier as the learning algorithm
eta0 is the learning rate, but it must explicitly be set to be constant
"""

# import general modules
import numpy                 as np

# import data, and functions specific to scikit-learn
from sklearn                 import datasets
from sklearn.linear_model    import SGDClassifier
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler


# import the Iris flower dataset
# Note that this is standard dataset included in scikit-learn
iris = datasets.load_iris()

# define our x-values and our y_values
    # the x-values signify the data that we will use to predict our y
        # in our case, this is the sepal/petal length/width
        # we will use these measures to predict the species
    # the y-values signify what we will predict
        # in our case, we aim to predict the species based on the x-values
X = iris.data
y = iris.target

# class_names is a translation from the numbers in y to the actual names
# these actual names will be mainly used when plotting
# y == 0 = setosa, y == 1 = versicolor, and y == 2 = virginica
class_names = iris.target_names
# split the data into a training set and a test set
# the dedicated function 'train_test_split()' is used to do this
# this splitting is random, hence the 'random_state'
# The number represents a 'random seed', which makes sure that the set is 
# split in exactly the same pieces for each run
# if the seed was deleted, the dataset would be split in different pieces 
# with each run
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y)

# train a scaler based on the training data
# this scaler normalizes all our features (our x values)
# this is accomplished by subtracting the mean and scaling to the variance
# normalization of a dataset is a common requirement for many machine 
# learning estimators: they might behave badly if the individual features do 
# not more or less look like standard normally distributed data 
# (e.g. Gaussian with 0 mean and unit variance)
# other techniques of machine learning often assume that your features are 
# normalized like we did here
# if this is not the case, then the machine learning will perform unexpected,
# and it might even happen that not all features are learned correctly
sc = StandardScaler()
sc.fit(X_train)

# apply the trained scaler to the X training data, and apply the same scaler to 
# the X test data
# by doing so, we normalize all our features, and we minimize the chance
# that our machine learning fails due to high variance in the dataset

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

classification_algorithm = SGDClassifier(max_iter = 20,
                                      verbose = 0, learning_rate = "constant",
                                      eta0 = .5)

# train our classification algorithm on the part of the data that we 
# reserved for the training process
classification_algorithm.fit(X_train_std, y_train)

# apply the trained classification algorithm on the test data to make 
# predictions
    # i.e. we make a prediction about y (which species is this) based on x
    # (the measurements) for all X in X_test_std
# Obviously, we hope that all our predictions match with the actual labels, 
# which are stored in y_test
y_pred = classification_algorithm.predict(X_test_std)

# view the predicted y and the actual y respectively
# print(y_pred)
# print(y_test)

# absolute number of times the classification was wrong
compared = np.array(y_pred == y_test)
absolute_wrong = (compared == False).sum()
print("Our classification was wrong for {0} out of the {1} cases.".format(absolute_wrong, len(compared)))

# view the accuracy of the model using a scikit-learn function, which is
# calculated by doing:
# 1 - (observations predicted wrong / total observations)
print('Accuracy percentage: {0:.2f}'.format(accuracy_score(y_test, y_pred) * 100))
