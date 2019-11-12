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
import itertools
import numpy                 as np
import pandas                as pd

from sklearn                 import datasets
from sklearn.linear_model    import Perceptron
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

#%%

# import the iris flower dataset
iris = datasets.load_iris()
X    = iris.data
y    = iris.target

#%%

# define where to store the simulation results
arr = np.zeros((50, 3))

# find all permutations
permutations = list(itertools.combinations('012', 2))

# 50 simulations for each comparison
for i in range(50):
    for perm in permutations:
        
        # apply boolean mask to get the values we need from the array
        first_class, second_class = int(perm[0]), int(perm[1])
        mask    = np.where((y == first_class) | (y == second_class))
        X_selec = X[mask]
        y_selec = y[mask]
    
        # split data in training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X_selec, 
                                                            y_selec)
        
        # classifier
        classification_algorithm = Perceptron(max_iter         = 100,
                                              tol              = 1e-3,
                                              verbose          = 0,
                                              n_iter_no_change = 5)
        
        # fit ('train') classifier to the training data
        classification_algorithm.fit(X_train, y_train)
        
        # predict y based on x for the test data
        y_pred = classification_algorithm.predict(X_test)
        
        # link the names of the families to the int label
        name_dict = {0: 'setosa',
                     1: 'versicolor',
                     2: 'virginica'}
        
        # store the accuracy in the pandas DataFrame
        arr[i, permutations.index(perm)] = accuracy_score(y_test, y_pred) * 100
       
#%%
    
# results of the simulation    
simulation_results         = pd.DataFrame(arr)
colnames                   = ['Set - Vers', 'Set - Virg', 'Vers - Virg']
simulation_results.columns = colnames

print('Mininum accuracy of Setosa vs Versicolor: {0:.2f} %'.format(np.min(simulation_results['Set - Vers'])))
print('Mininum accuracy of Setosa vs Virginica: {0:.2f} %'.format(np.min(simulation_results['Set - Virg'])))
print('Mininum accuracy of Versicolor vs Virginica: {0:.2f} %'.format(np.min(simulation_results['Vers - Virg'])))
