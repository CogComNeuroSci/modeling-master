#!/usr/bin/python3
# -*- coding: utf-8 -*-


'''
code by Pieter Huycke and Tom Verguts
model with two hidden layers for Stroop task
'''


import numpy                      as     np
from   sklearn.metrics            import accuracy_score
from   sklearn.neural_network     import MLPClassifier


#%%
'''
Define vectors
'''

# 'focus on color'
red_red_color     = [0, 1, 0, 1, 0, 1] # Answer should be: 'red'
green_green_color = [0, 1, 1, 0, 1, 0] # Answer should be: 'green'
red_green_color   = [0, 1, 0, 1, 1, 0] # Answer should be: 'red'
green_red_color   = [0, 1, 1, 0, 0, 1] # Answer should be: 'green'

# 'focus on the word'
red_red_word      = [1, 0, 0, 1, 0, 1] # Answer should be: 'red'
green_green_word  = [1, 0, 1, 0, 1, 0] # Answer should be: 'green'
red_green_word    = [1, 0, 0, 1, 1, 0] # Answer should be: 'green'
green_red_word    = [1, 0, 1, 0, 0, 1] # Answer should be: 'red'

response_1, response_2, response_3, response_4 =  [1], [-1],  [1], [-1]
response_5, response_6, response_7, response_8 =  [1], [-1], [-1],  [1]

# Bundle them together
all_possible_inputs  = np.array([red_red_color, green_green_color, red_green_color, green_red_color,
                                 red_red_word,  green_green_word,  red_green_word,  green_red_word])
all_possible_outputs = np.array([response_1, response_2, response_3, response_4, 
                                 response_5, response_6, response_7, response_8])

# Amount of possible inputs
length_inputs = len(all_possible_inputs)
    
# Repeat every possible input pattern 50 times
inputted_patterns    = np.tile(all_possible_inputs,  (5, 1))
outputs              = np.ravel(np.tile(all_possible_outputs, (5, 1)))

#%%
np.set_printoptions(precision=2)
# note I make no train/test distinction now
X_train = inputted_patterns
y_train = outputs

mlp = MLPClassifier(hidden_layer_sizes=(2,2), 
                    max_iter=10000,
                    solver='sgd', 
                    verbose=0)

# fit  classifier to the training data
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_train)

print("accuracy with 2 hidden layers = {:.2%}".format(accuracy_score(y_train, y_pred)))