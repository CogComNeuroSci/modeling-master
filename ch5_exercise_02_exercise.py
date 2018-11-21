#!/usr/bin/python3
# -*- coding: utf-8 -*-


'''
@author: Pieter
Pieter.Huycke@UGent.be

Code adapted from the solution of test 01
Credit to both Tom and Mehdi

- - - - - - - - - - - - 

Stuck using a function?
No idea about the arguments that should be defined?

Type:
help(function_name)
to let Python help you!
'''


import numpy                      as     np

from   ch5_plotting_helper        import plot_confusion_matrix
from   sklearn.linear_model       import Perceptron
from   sklearn.metrics            import confusion_matrix, accuracy_score
from   sklearn.model_selection    import train_test_split
from   sklearn.neural_network     import MLPClassifier


#%%
'''
* GENERAL EXPLANATION *

In this code, we are going to train a simple Stroop model
You are going to implement this yourself.
However, as we are aware that the learning curve might be a bit steep, we will
provide some hints to get you started.

The paradigm of a Stroop task is simple:
A participant sits in front of a computerscreen, and sees a sequence of English
colorwords (in written format). A cue presented prior to the word tells the 
participant to what dimension one should respond. Two response dimensions are 
available: the word itself, and the color of the word.
If the color of the word and the written format match (e.g. RED written in a 
red color), then a congruency effect will occur. Alternatively, if color and 
word do not match (e.g. GREEN written in red), participants become slower.

To keep things simple, we will only look at two words and two colors.
You can imagine this as a Stroop task where participants see the words 'GREEN'
and 'RED', written in the colors green and red.
The cue that indicates to which dimension the subjects have to respond is also
binomial: a subject should either react to the word itself, or to the color.

In other words, we have three dimensions that can take two different values:
    - color of the word
    - written form
    - cue that represents the relevant dimension
Given this, we know that we will need 6 input units (3 dimensions with each
two possible values).
By doing so, we can represent every possible input that our participant might
see.

How many output units do we need?
Know that we have two response dimensions: "It's green" and "It's red"
What the participant thinks is often measured by key presses (e.g. press right
if the answer is 'green').
Therefore, we will need to output units: to represent every possible outcome

For convenience, we will fix the cue units: this means that the participants 
always have to respond to the same stimulus dimension.
Fixing the cue unit is the same as me saying to the participant 'you should
only respond to the color of the shown word.'
'''

#%%
'''
* ONLY ONE RELEVANT STIMULUS DIMENSION *

Define our model and the train data
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
# An example to illustrate this:
# The following pattern:
    # [0, 1, 0, 1, 0, 1]
# might represent the word 'RED', written in red, and the participant has to 
# (according to the cue) respond to the color of the word
# Alternatively, the pattern
    # [0, 1, 1, 0, 1, 0]
# might represent the word 'GREEN', written in green, but the participant still
# has to respond to the color of the word
    
# Below, we will first define all the possible input patterns, keeping in mind
# that the cue never changes (i.e. the participants also should react to one
# specific dimension of the shown stimulus).
red_red     = ...
green_green = ...
red_green   = ...
green_red   = ...

response_1, response_3 = ...
response_2, response_4 = ...

# Bundle them together
all_possible_inputs  = ...
all_possible_outputs = ...

# Amount of possible inputs
...
    
# Repeat every possible input pattern 50 times
inputted_patterns    = np.tile(...,  (... * ..., ...))
outputs              = np.ravel(np.tile(..., (... * ..., ...)))

# The strings associated with our possible outcomes
# Here, only two labels are possible ('green', and 'red')
class_names = ['Color relevant', 'Word relevant']

#%%
'''
* DELTA LEARNING *

Train the model using the data we created prior to this step
This is a well-known process, so we will let you fill in the rest
Use a Perceptron object to perform delta learning
You can use 100 learning cycles
'''

# split data in training and testing set
...

# define classifier (Perceptron object from scikit-learn)
# use a random state of '2020'
...

# fit ('train') classifier to the training data
...

'''
* PLOTTING  *

Test the model, show the confusion matrix.
We should note that the accuracy is 100%.
'''
# predict y based on x for the test data
...

# select wrong predictions (absolute vals) and print them
...
...
print("Our classification was wrong for {0} out of the {1} cases.". ...)


# print accuracy using dedicated function
print('Accuracy percentage: {0:.2f}'. ...)
# define confusion matrix and set numpy precision to two numbers
...
np.set_printoptions(precision=2)

# plot absolute confusion matrix
# do not normalize
# think of a descriptive plot title to show with your plot
...

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
red_red_color     = ...
green_green_color = ...
red_green_color   = ...
green_red_color   = ...

# 'focus on the word'
red_red_word      = ...
green_green_word  = ...
red_green_word    = ...
green_red_word    = ...

response_1, response_2, response_3, response_4 =  ...
response_5, response_6, response_7, response_8 =  ...

# Bundle them together
all_possible_inputs  = ...
all_possible_outputs = ...

# Amount of possible inputs
...
    
# Repeat every possible input pattern 50 times
inputted_patterns    = np.tile(...,  (... * ..., ...))
outputs              = np.ravel(np.tile(..., (... * ..., ...)))

#%%
'''
* DELTA LEARNING *

Train the model using the data we created prior to this step
Again, you can use 100 learning cycles
'''

# split data in training and testing set
...

# define classifier (Perceptron object from scikit-learn)
...

# fit ('train') classifier to the training data
...

'''
* PLOTTING  *

Test the model, show the confusion matrix.
We should note that the model performs way worse than before.
'''
# predict y based on x for the test data
...

# select wrong predictions (absolute vals) and print them
...
...
print("Our classification was wrong for {0} out of the {1} cases.". ...)


# print accuracy using dedicated function
print('Accuracy percentage: {0:.2f}'. ...)

# define confusion matrix and set numpy precision to two numbers
...
np.set_printoptions(precision=2)

# plot absolute confusion matrix
# do not normalize
# think of a descriptive plot title to show with your plot
...

#%%
'''
* TWO RELEVANT STIMULUS DIMENSIONS *
* BACKPROPAGATION *

Train the model that incorporated different tasks
Now, implemented an extra hidden layer with 4 units, and use backpropagation
to alter the weight matrix
'''

# split data in training and testing set
...

# define the multilayer perceptron learner
# (search on Google!)
# you should have:
    # 1 hidden layer with 4 units
    # 500 learning iterations
    # use 'stochastic gradient descent' (google if needed) as optimization algorithm
    # use a logistic activation function
    # use a start learning rate of 0.1
mlp = MLPClassifier(hidden_layer_sizes=..., 
                    max_iter=...,
                    solver=..., 
                    verbose=0,
                    random_state=2020,
                    learning_rate_init=..., 
                    ...=...)

# fit  classifier to the training data
...

'''
* PLOTTING  *

Test the model, show the confusion matrix.
We should note that the model performs way worse than before.
'''
# predict y based on x for the test data
...

# select wrong predictions (absolute vals) and print them
...
...
print("Our classification was wrong for {0} out of the {1} cases.". ...)


# print accuracy using dedicated function
print('Accuracy percentage: {0:.2f}'. ...)

# define confusion matrix and set numpy precision to two numbers
...
np.set_printoptions(precision=2)

# plot absolute confusion matrix
# do not normalize
# think of a descriptive plot title to show with your plot
...
