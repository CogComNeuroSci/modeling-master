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
import numpy                 as np
import seaborn               as sns

from sklearn                 import datasets
from sklearn.linear_model    import Perceptron
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

#%%

# import the Iris flower dataset
iris        = datasets.load_iris()
X           = iris.data
y           = iris.target

#%%

sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g    = sns.pairplot(iris, 
                    hue     = "species", 
                    markers = ["o", "s", "D"],
                    palette = sns.color_palette('colorblind'))