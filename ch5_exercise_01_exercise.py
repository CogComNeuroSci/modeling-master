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


# import relevant modules
import ch0_delta_learning      as     delta_learning
import itertools
import matplotlib.pyplot       as     plt
import numpy                   as     np

from   sklearn.linear_model    import Perceptron
from   sklearn.model_selection import train_test_split


#%%
'''
* Defining the dataset *

In this part, we will define the data we are going to train with.
We will use a dataset you are familiar with to boost your understanding:
the dataset that you defined in test 01.
By doing so, you don't have to spend a lot of time trying to understand the 
data and it's implications, as you are already familiar with it.

This code is (almost entirely) adopted from the first part of the test.
'''

# We define our learning parameter
beta = .1

# Training samples (activation (x) of each input unit)
# Use 40 of each
cat_proto = np.array([0, 1, 1])
n_train_cats = ...
dog_proto = np.array([1, 1, -1])
n_train_dogs = ...

# Noise on exemplars
std_noise = 0.01

# Make a large array with all the examples of 'cats' and 'dogs'
# that are 'shown' to the model
train_samples = cat_proto
for loop in range(n_train_cats-1):
    train_samples = np.vstack((train_samples, cat_proto))
for loop in range(n_train_dogs):
    train_samples = np.vstack((train_samples, dog_proto))
    
noise = np.random.randn(n_train_cats+n_train_dogs, 3)*std_noise
train_samples = train_samples + noise

# The targets (basically representing "dog" or "cat"):
#                   cat is represented by [1, 0] 
#                   dog is represented by [0, 1]
targets = np.array( [1, 0] )
for loop in range(n_train_cats-1):
    targets = np.vstack((targets, [1, 0]))
for loop in range(n_train_dogs):
    targets = np.vstack((targets, [0, 1]))


#%%
'''
* Learning through Delta learning *

In this part of the code, we will use Delta learning to learn to association
between specific patterns and the concepts of cats and dogs.
This is really similar to the idea of associating a song of a music band. and 
their picture group, so if that was clear then this will also be clear!
'''

# Define a weight matrix exclusively filled with zeros
# Make a weight matrix that fits the data you are working with
weight_matrix = delta_learning.initialise_weights(..., 
                                                  ..., 
                                                  ...,
                                                  ..., 
                                                  ...)

# Show me what you got 
print('Our original weight matrix, for now filled with zeros:\n', weight_matrix)

# Make a copy of the original weight matrix
...

# Activation associated with the all zero weight matrix
activation_original = ...
print('\nActivation levels at output for the original weight matrix:\n', activation_original)

# Change the original weight matrix towards the desired weight matrix
    # i.e. the weight matrix for which the output is really close to the 
    # desired output (in our case, the desired output can vary:
        # in the case of a cat, the model should output [1, 0]
        # in the case of a dog, the model should output [0, 1]
# Let the model learn through Delta learning
# Show *each* example of a cat or dog 100 times to the model, but do not forget
# to show *every* example of a cat or dog that is available!
# Mind that we defined a learning rate earlier ... 
    
times_shown = ...

# define a random order
...

# loop over the data in a random fashion
for ... in range(...):
    for ... in np.arange(1, ... + 1):
        weights_after_learning = delta_learning.weight_change(...,
                                                              ...,
                                                              ...,
                                                              ...)
        ... = weights_after_learning


print('\nOur altered weight matrix after {0} trials of delta learning:\n{1}').format(...)

#%%
'''
* testing after Delta learning *

Test your model, represent a cat example and a dog example to the model
See whether the model's output resembles the desired activation
'''

print('\n*\n* Testing model performance with the "home-made" form of Delta learning\n*\n')

# Showing a cat to the model after performing Delta learning
activation_after_seeing_cat = ...
print('Activation levels after seeing a cat:', 
      np.round(activation_after_seeing_cat, 3))

# Showing a dog to the model after performing Delta learning
activation_after_seeing_dog = ...
print('Activation levels after seeing a dog:', 
      np.round(activation_after_seeing_dog, 3))

#%%
'''
* Learning using scikit-learn *

In this part of the code, we will use scikit-learn to learn to association
between specific patterns and the concepts of cats and dogs.
Similar to exercise 3 of the previous course, we are going to define a 
Perceptron object.
Please remember that the Perceptron basically does Delta learning at its core.
Here, we would expect that the output would be very similar to the results we
obtained with the code above...
'''

# Define new targets:
    # make sure that the array  [1]  represents a cat
    # make sure that the array [-1] represents a dog
targets = np.array(...)
for loop in range(...):
    targets = np.vstack((...))
for loop in range(n_train_dogs):
    targets = np.vstack((...)))
# Make a single array of the (for now) double array
    # Print before and after if you don't understand what happens! 
targets = np.ravel(targets)

# split data in training and testing set
X_train, X_test, y_train, y_test = ...

# define the classifier (Perceptron object from scikit-learn)
    # 100 cycles
    # No random state
    # Verbose set to False
classification_algorithm = ...

# fit ('train') classifier to the training data (no scaling is required)
classification_algorithm.fit(...)

#%%
'''
* testing after Delta learning using the Perceptron *

Test your model, represent a cat example and a dog example to the model
See whether the model's output resembles the desired activation
'''

print('\n*\n* Testing model performance with the Perceptron from scikit-learn\n*\n')

# Show a cat to the model, and see that the model predicts based on the input
# Remember that in this case, the output:
    # 'cat' is represented by [ 1]
    # 'dog' is represented by [-1]
reaction_to_cat, reaction_to_dog = ..., \
                                   ...

# Presenting a cat
print('We represented the following pattern (which is known to be a cat):\n', ...)
if ...:
    print('After learning, the model labels this as a cat')
else:
    print('After learning, the model labels this as a dog')

# Presenting a dog
print('\nWe represented the following pattern (which is known to be a dog):\n',...)
if ...:
    ...


print(.. the conclusion of our little experiment)

