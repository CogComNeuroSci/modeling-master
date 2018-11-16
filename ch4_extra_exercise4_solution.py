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