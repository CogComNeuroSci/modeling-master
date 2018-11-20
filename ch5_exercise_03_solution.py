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
*The biased model*

In this code, we are going to recreate the Stroop model.
Here, the idea is that a human is biased towards reading words.
Because of this, we have to bias the model towards reading words.
To do this, we have to present the 'word naming task' more than the 
'color naming task'
This approach is similar to previous work where we presented more dogs than 
cats when training our model

To keep track of our model, we will first have to assign interpretations to
our unit activations (like we did in the previous exercise).
Then, we can define what each pattern represents, and what response is 
appropriate for each pattern.
Remember to learn the 'word task' more to install the bias towards reading

Then, we want to train our model with the data, which leads to a model that 
is able to complete the task, but has a bias towards reading vs naming the color
'''

# Unit interpretation:
    # units 1 and 2 represent the cue that is given (this one is fixed)
        # [0, 1] = 'color' is relevant dimension
        # [1, 0] = 'word' is relevant dimension
    # units 3 and 4 represent the color of the word:
        # [1, 0] = green
        # [0, 1] = red
    # units 5 and 6 represent the written format:
        # [1, 0] = green
        # [0, 1] = red
    # response units:
        #  [1] if the response should be 'red'
        # [-1] if the response should be 'green'

# task: 'name the color of the presented word'
red_red_color     = [0, 1, 0, 1, 0, 1] # Answer should be: 'red'
green_green_color = [0, 1, 1, 0, 1, 0] # Answer should be: 'green'
red_green_color   = [0, 1, 0, 1, 1, 0] # Answer should be: 'red'
green_red_color   = [0, 1, 1, 0, 0, 1] # Answer should be: 'green'

# task: 'read the presented word'
red_red_word      = [1, 0, 0, 1, 0, 1] # Answer should be: 'red'
green_green_word  = [1, 0, 1, 0, 1, 0] # Answer should be: 'green'
red_green_word    = [1, 0, 0, 1, 1, 0] # Answer should be: 'green'
green_red_word    = [1, 0, 1, 0, 0, 1] # Answer should be: 'red'

# appropriate responses for each case
response_1, response_2, response_3, response_4 =  [1], [-1],  [1], [-1]
response_5, response_6, response_7, response_8 =  [1], [-1], [-1],  [1]

# bundle them together
color_inputs  = np.array([red_red_color, green_green_color, 
                          red_green_color, green_red_color])
word_inputs   = np.array([red_red_word,  green_green_word,  
                          red_green_word,  green_red_word])

# make a copy of the original arrays, because I believe you might want to use 
# those later on...
color_inputs_copy, word_inputs_copy = np.copy(color_inputs), \
                                      np.copy(word_inputs)

# define the appropriate responses for the 'name the color' task
color_outputs = np.array([response_1, response_2, response_3, response_4])

# define the appropriate responses for the 'read the word' task
word_outputs  = np.array([response_5, response_6, response_7, response_8])

# again make a copy of the original arrays
color_outputs_copy, word_outputs_copy = np.copy(color_outputs), \
                                        np.copy(word_outputs)

# amount of possible inputs (the same for the word task of course)
length_inputs = len(color_inputs)

# repeat the input patterns (repeat the word task more, for the bias)
color_inputs   = np.tile(color_inputs, (length_inputs * 2, 1))
word_inputs    = np.tile(word_inputs,  (length_inputs * 7, 1))

# repeat the responses the same amount of times: our responses should match
# our defined input patterns
color_outputs  = np.tile(color_outputs, (length_inputs * 2, 1))
word_outputs   = np.tile(word_outputs,  (length_inputs * 7, 1))

# stack the input patterns and their associated responses
# we use ravel because scikit learn will complain (but still work) when ravel
# is not used
inputted_patterns  = np.vstack((color_inputs, word_inputs))
outputs            = np.ravel(np.vstack((color_outputs, word_outputs)))

# the strings associated with our possible outcomes
# here, only two labels are possible ('color' and 'word')
# this because we have two different relevant dimensions: the color dimension
# and the word dimension
class_names = ['Color', 'Word']

#%%
'''
* Fitting the model *

In this part, we use a multilayer perceptron to train our model
Similar to the previous exercise, we will use a hidden layer with 4 hidden 
units
Our learning rate, the random_state, ... and other parameters remain constant
'''

# define MultiLayerPerceptron
mlp = MLPClassifier(hidden_layer_sizes=(4,), 
                    max_iter=100,
                    solver='sgd', 
                    verbose=0,
                    random_state=2020,
                    learning_rate_init=.01, 
                    activation='logistic')

# train the model based on the data
mlp.fit(inputted_patterns, outputs)

#%%
'''
* Testing on the congruent trials *

Now, we will use our trained model to perform the Stroop task, but only for the
congruent trials. 

In this case, a congruent trial is when the ink color is the 
same as the word (i.e., green written in green ink) and an incongruent trial is
when the ink color is different to the word meaning 
(i.e., green written in red ink)
'''

color_congr = np.tile(color_inputs_copy[0:2], (length_inputs * 10, 1))
word_congr  = np.tile(word_inputs_copy[0:2], (length_inputs * 10, 1))

congruent   = np.vstack((color_congr, word_congr))

color_outputs_congr = np.tile(color_outputs_copy[0:2], (length_inputs * 10, 1))
word_outputs_congr  = np.tile(word_outputs_copy[0:2], (length_inputs * 10, 1))

output_congruent    = np.ravel(np.vstack((color_outputs_congr, word_outputs_congr)))

# predict y based on x for the test data
y_pred = mlp.predict(congruent)

# print accuracy using dedicated function
print('Accuracy percentage in the congruent trials: {0:.2f}%'.format(accuracy_score(output_congruent, y_pred) * 100))

#%%
'''
* Testing on the incongruent trials *

Now, we will use our trained model to perform the Stroop task, but only for the
incongruent trials. 

The same definitions with respect to (in)congruency hold
'''

color_incongr = np.tile(color_inputs_copy[2:4], (length_inputs * 10, 1))
word_incongr  = np.tile(word_inputs_copy[2:4], (length_inputs * 10, 1))

incongruent   = np.vstack((color_incongr, word_incongr))

color_outputs_incongr = np.tile(color_outputs_copy[2:4], (length_inputs * 10, 1))
word_outputs_incongr  = np.tile(word_outputs_copy[2:4], (length_inputs * 10, 1))

output_incongruent    = np.ravel(np.vstack((color_outputs_incongr, word_outputs_incongr)))

# predict y based on x for the test data
y_pred = mlp.predict(incongruent)

# print accuracy using dedicated function
print('Accuracy percentage in the incongruent trials: {0:.2f}%'.format(accuracy_score(output_incongruent, y_pred) * 100))

#%%
'''
Change the weights
'''
np.random.seed(2020)

original_weights_to_hidden = np.copy(mlp.coefs_[0])

for i in range(100):
      
    # predict y based on x for the test data
    y_pred = mlp.predict(inputted_patterns)
    
    # print accuracy using dedicated function
    print('Cycle {0:3d}: Accuracy percentage: {1:.2f}'.format(i, accuracy_score(outputs, y_pred) * 100))
    if (accuracy_score(outputs, y_pred)*100) < 80:
        print('Accuracy droppped below 80% in cycle {0:3d}'.format(i))
        
        cnf_matrix = confusion_matrix(outputs, y_pred)
        np.set_printoptions(precision=2)
    
        plot_confusion_matrix(cnf_matrix, 
                          classes = class_names,
                          normalize = False,
                          plot_title = 'Confusion matrix for the Stroop task\n' \
                                       'Weights from input layer to hidden layer were distorted')
        break
    
    mlp.coefs_[0] += (np.random.randn(6, 4) / 25)