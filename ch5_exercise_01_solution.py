#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
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
"""


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
cat_proto = np.array([0, 1, 1])
n_train_cats = 40
dog_proto = np.array([1, 1, -1])
n_train_dogs = 40

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
weight_matrix = delta_learning.initialise_weights(train_samples[0], 
                                                  targets[0], 
                                                  zeros=True,
                                                  predefined=False, 
                                                  verbose=True)

# Show me what you got 
print('Our original weight matrix, for now filled with zeros:\n', weight_matrix)

# Make a copy of the original weight matrix
original_weight_matrix = np.copy(weight_matrix)

# Activation associated with the all zero weight matrix
activation_original = delta_learning.internal_input(train_samples[0],
                                                    weight_matrix)[0]
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
    
times_shown = 100

for indices in range(train_samples.shape[0]):
    for loop_var in np.arange(1, times_shown + 1):
        weights_after_learning = delta_learning.weight_change(beta,
                                                              train_samples[indices],
                                                              targets[indices],
                                                              weight_matrix)
        weight_matrix = weights_after_learning


print('\nOur altered weight matrix after {} trials of delta learning:\n'.format(times_shown), 
      weight_matrix)

#%%
'''
* testing after Delta learning *

Test your model, represent a cat example and a dog example to the model
See whether the model's output resembles the desired activation
'''

print('\n*\n* Testing model performance with the "home-made" form of Delta learning\n*\n')

# Showing a cat to the model after performing Delta learning
activation_after_seeing_cat = delta_learning.internal_input(train_samples[0],
                                                          weight_matrix)[0]
print('Activation levels after seeing a cat:', 
      np.round(activation_after_seeing_cat, 3))

# Showing a dog to the model after performing Delta learning
activation_after_seeing_dog = delta_learning.internal_input(train_samples[-1],
                                                          weight_matrix)[0]
print('Activation levels after seeing a dog:', 
      np.round(activation_after_seeing_dog, 3))

#%%
'''
* Learning through backpropagation *

In this part of the code, we will use backpropagation to learn to association
between specific patterns and the concepts of cats and dogs.
This code relies on scikit-learn (remember the Iris dataset exercise)!
Similar to that exercise, we are going to define a Perceptron object.
Please remember that the Perceptron basically does Delta learning at its core.
Here, we would expect that the output would be very similar to the results we
obtained with the code above...
'''

# Define new targets:
    # make sure that the array [1]  represents a cat
    # make sure that the array -[1] represents a dog
targets = np.array(1)
for loop in range(n_train_cats-1):
    targets = np.vstack((targets, [1]))
for loop in range(n_train_dogs):
    targets = np.vstack((targets, [-1]))
# Make a single array of the (for now) double array
    # Print before and after if you don't understand what happens!
targets = np.ravel(targets)

# split data in training and testing set
X_train, X_test, y_train, y_test = train_test_split(train_samples, 
                                                    targets)

# define classifier (Perceptron object from scikit-learn)
classification_algorithm = Perceptron(max_iter = 100,
                                      verbose = 0)

# fit ('train') classifier to the training data
classification_algorithm.fit(X_train, y_train)

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
reaction_to_cat, reaction_to_dog = classification_algorithm.predict([train_samples[ 0]]), \
                                   classification_algorithm.predict([train_samples[-1]])

# Presenting a cat
print('We represented the following pattern (which is known to be a cat):\n', train_samples[0])
if reaction_to_cat[0] == 1:
    print('After learning, the model labels this as a cat')
else:
    print('After learning, the model labels this as a dog')

# Presenting a dog
print('\nWe represented the following pattern (which is known to be a dog):\n', train_samples[-1])
if reaction_to_dog[0] == 1:
    print('After learning, the model labels this as a cat')
else:
    print('After learning, the model labels this as a dog')


print('\n\n**\nBoth models are able to differentiate cats from dogs! *\n**')

