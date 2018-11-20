#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 

Stuck using a function?
No idea about the arguments that should be defined?

Type:
help(function_name)
to let Python help you!
"""


import numpy                      as     np

from   plotting_helper            import plot_confusion_matrix
from   sklearn.metrics            import confusion_matrix, accuracy_score
from   sklearn.neural_network     import MLPClassifier

#%%
'''
* TWO RELEVANT STIMULUS DIMENSIONS *

In the following, we will create data for a model where the participant has to
switch tasks. 
In other words, the first two units will not have different activation patterns
(while they were fixed previously).
Try to use Delta learning to configure a correct set of weights.
See whether this works out.
If not, use backpropagation to configure a set of weights which can solve the 
problem.
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

# bundle them together
color_inputs  = np.array([red_red_color, green_green_color, red_green_color, green_red_color])
word_inputs   = np.array([red_red_word,  green_green_word,  red_green_word,  green_red_word])

# make a copy of the original arrays, because I believe you might want to use 
# those later on...
color_inputs_copy, word_inputs_copy = np.copy(color_inputs), \
                                      np.copy(word_inputs)

# define
color_outputs = np.array([response_1, response_2, response_3, response_4])
word_outputs  = np.array([response_5, response_6, response_7, response_8])

# again make a copy of the original arrays
color_outputs_copy, word_outputs_copy = np.copy(color_outputs), \
                                        np.copy(word_outputs)

# amount of possible inputs
length_inputs = len(color_inputs)

# repeat every possible input pattern 50 times
color_inputs   = np.tile(color_inputs, (length_inputs * 2, 1))
word_inputs    = np.tile(word_inputs,  (length_inputs * 7, 1))

color_outputs  = np.tile(color_outputs, (length_inputs * 2, 1))
word_outputs   = np.tile(word_outputs,  (length_inputs * 7, 1))

inputted_patterns  = np.vstack((color_inputs, word_inputs))
outputs            = np.ravel(np.vstack((color_outputs, word_outputs)))

# the strings associated with our possible outcomes
# here, only two labels are possible ('green', and 'red')
class_names = ['Color', 'Word']

#%%
'''
Fit the model to the data that is available
'''

# define MultiLayerPerceptron
mlp = MLPClassifier(hidden_layer_sizes=(4,), 
                    max_iter=105,
                    solver='sgd', 
                    verbose=0,
                    random_state=2020,
                    learning_rate_init=.01, 
                    activation='logistic')

# train the model based on the data
mlp.fit(inputted_patterns, outputs)

#%% 
'''
Test the model on a balanced dataset
In this dataset, the amount of cues that require a 'word' response is equal
to the amount of stimuli that requires a 'color response'.
'''

color_inputs    = np.tile(color_inputs_copy, (length_inputs * 5, 1))
word_inputs     = np.tile(word_inputs_copy,  (length_inputs * 5, 1))

color_outputs   = np.tile(color_outputs_copy, (length_inputs * 5, 1))
word_outputs    = np.tile(word_outputs_copy,  (length_inputs * 5, 1))

balanced_input  = np.vstack((color_inputs, word_inputs))
balanced_output = np.ravel(np.vstack((color_outputs, word_outputs)))

# predict y based on x for the test data
y_pred = mlp.predict(balanced_input)

print(mlp.coefs_[0])

# select wrong predictions (absolute vals) and print them
compared = np.array(y_pred == balanced_output)
absolute_wrong = (compared == False).sum()
print("Our classification was wrong for {0} out of the {1} cases.".format(absolute_wrong, len(compared)))

# print accuracy using dedicated function
print('Accuracy percentage: {0:.2f}'.format(accuracy_score(balanced_output, y_pred) * 100))

# define confusion matrix and set numpy precision to two numbers
cnf_matrix = confusion_matrix(balanced_output, y_pred)
np.set_printoptions(precision=2)

# plot absolute confusion matrix
plot_confusion_matrix(cnf_matrix, 
                      classes = class_names,
                      normalize = False,
                      plot_title = 'Confusion matrix for the Stroop task\nWeights from input layer to hidden layer were distorted')


#%%
'''
Change the weights
'''
np.random.seed(2020)

original_weights_to_hidden = np.copy(mlp.coefs_[0])
mlp.coefs_[0]              = mlp.coefs_[0] + (np.random.randn(6, 4) / 10)

# predict y based on x for the test data
y_pred = mlp.predict(balanced_input)

# select wrong predictions (absolute vals) and print them
compared = np.array(y_pred == balanced_output)
absolute_wrong = (compared == False).sum()
print("Our classification was wrong for {0} out of the {1} cases.".format(absolute_wrong, len(compared)))


# print accuracy using dedicated function
print('Accuracy percentage: {0:.2f}'.format(accuracy_score(balanced_output, y_pred) * 100))

# define confusion matrix and set numpy precision to two numbers
cnf_matrix = confusion_matrix(balanced_output, y_pred)
np.set_printoptions(precision=2)

# plot absolute confusion matrix
plot_confusion_matrix(cnf_matrix, 
                      classes = class_names,
                      normalize = False,
                      plot_title = 'Confusion matrix for the Stroop task\nWeights from input layer to hidden layer were distorted')
