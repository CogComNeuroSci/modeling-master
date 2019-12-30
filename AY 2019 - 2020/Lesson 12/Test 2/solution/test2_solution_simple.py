    #!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke

Question 3:
    A two-layer model is not able to distinguish grammatical from
    ungrammatical sentences (Perceptron performs at around 50% acccuracy).
    This shows that this grammar rule is not a linearly separable problem.
    Whether we use a bias unit does not change much: 51.5% accuracy without a
    bias unit and 47.7% accuracy with a bias unit.
    The bias unit does not change much in accuracy which confirms that the
    problem is non-linearly separable since it does not increase accuracy
    whether the separating plane (line in 3 dimensions) passes through the
    origin or not.
    
Question 5:
    A multi-layer perceptron (MLP) with 1 hidden layer is able to perform the
    task with a minimum of 5 hidden units: 98.78% accuracy
    This model performs better than the perceptron because it is able to
    classify non-linearly separated data (i.e. classes).

Questino 6:
    The MLP with 2 hidden layers but the same amount of units overall (5 hidden
    units: 3 in the first layer, 2 in the second layer) performs worse than the
    1 hidden layer MLP ((5) hidden units: 98.78% accuracy, (3, 2) hidden units:
    73.42%). This shows that to classify grammatical versus ungrammatical
    sentences the model requires a hidden layer with at least 5 units,
    therefore when we split these units between two layers it is not able to
    perform as well.

"""

#%%

# import modules
import itertools
import numpy as np

from sklearn.linear_model    import Perceptron
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

#%%

SIMULATIONS = 40
REPETITIONS = 50
MAX_HIDDEN  = 20

#%%

# make all combinations and assign to an array or list (situation different)
combinations     = list(itertools.product([0, 1], repeat = 3))
separate_strings = []
grammar_coding   = np.zeros(len(combinations))

neural_coding    = {0 : [0, 1],  # A, coded as 0, with pattern [0,1]
                    1 : [1, 0]}  # B, coded as 1, with pattern [1,0]


# loop over all combinations; join strings and add coding
for indx in range(len(combinations)):
    
    # a temporary array to map '0' (A) to [0, 0, 0, 1]
    temp = []
    for arrs in combinations[indx]:
        temp.append(neural_coding.get(arrs))
    # glue the final code and save
    separate_strings.append(np.ravel(temp))
    
    # coding depending on grammar or not
    if temp[0] == temp[-1]:
        grammar_coding[indx] = 1
    else:
        grammar_coding[indx] = 0

# make separate_strings an array
separate_strings = np.array(separate_strings)

del combinations, indx, arrs, temp, neural_coding

#%%
        
# make an elaborate training -, and test set
X = np.repeat(separate_strings, REPETITIONS, axis = 0)
y = np.repeat(grammar_coding, REPETITIONS)

del grammar_coding, separate_strings

#%%

print('\n- - - -\nPerceptron\n- - - -\n')

#%%

# ---------- #
# PERCEPTRON #
# ---------- #

perceptron_acc = np.zeros(shape = (2, SIMULATIONS))

for with_bias_unit in [False, True]:

    for i in range(SIMULATIONS):
        
        # train test split (automatically randomizes the stimuli)
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y,
                                                            train_size = .6)
        
        # define classifier (Perceptron object from scikit-learn)
        classification_algorithm = Perceptron(max_iter         = 10000,
                                              tol              = 1e-3,
                                              verbose          = 0,
                                              fit_intercept    = with_bias_unit)
        
        
        # fit ('train') classifier to the training data
        classification_algorithm.fit(X_train, y_train)
        
        # predict y based on x for the test data
        y_pred = classification_algorithm.predict(X_test)
        
        perceptron_acc[int(with_bias_unit), i] = accuracy_score(y_test, y_pred) * 100

print('Average accuracy of our Perceptron without bias unit: {0:.2f}%\n'.format(np.mean(perceptron_acc[0, :])))
print('Average accuracy of our Perceptron with bias unit: {0:.2f}%\n'.format(np.mean(perceptron_acc[1, :])))

#%%

print('- - - -\nMulti-layered Perceptron (1 hidden layer)\n- - - -\n')

#%%

# --- #
# MLP #
# --- #

threshold_accuracy = 95

mlp_acc = np.zeros(SIMULATIONS)

for hidden_units in range(1, MAX_HIDDEN + 1):
    
    unsatisfied = False
    
    for loop_number in range(SIMULATIONS):
        # train test split (automatically randomizes the stimuli)
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y,
                                                            train_size = .6)
        
        # define classifier (Perceptron object from scikit-learn)
        classification_algorithm = MLPClassifier(hidden_layer_sizes = (hidden_units, ),
                                                 max_iter           = 10000, 
                                                 n_iter_no_change   = 10)
        
        # fit ('train') classifier to the training data
        classification_algorithm.fit(X_train, y_train)
        
        # predict y based on x for the test data
        y_pred = classification_algorithm.predict(X_test)

        # store accuracy for this simulation
        mlp_acc[loop_number] = accuracy_score(y_test, y_pred) * 100

    # print accuracy using a built-in sklearn function
    overall_accuracy = np.mean(mlp_acc)
    unsatisfied = np.mean(mlp_acc) < threshold_accuracy
    print('{0:.2f} % accuracy with {1:.0f} hidden unit(s):'.format(overall_accuracy, hidden_units))
    if unsatisfied:
        print('\t-> unsatisfactory\n')
    else:
        print('\t-> satisfactory\n')
        break


#%%

print('- - - -\nMulti-layered Perceptron (2 hidden layers)\n- - - -\n')

#%% MLP with 2 hidden layers
# The previous step indicated that 5 hidden units are enough so for question 6
# we will use 2 hidden layers with 3 and 2 hidden units, respectively.
layer1_hidden_units = 3
layer2_hidden_units = 2

mlp2_acc = np.zeros(SIMULATIONS)

for loop_number in range(SIMULATIONS):
    # train test split (automatically randomizes the stimuli)
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        train_size = .6)
    
    # define classifier (Perceptron object from scikit-learn)
    classification_algorithm = MLPClassifier(hidden_layer_sizes = (layer1_hidden_units, layer2_hidden_units),
                                             max_iter           = 10000, 
                                             n_iter_no_change   = 10)
    
    # fit ('train') classifier to the training data
    classification_algorithm.fit(X_train, y_train)
    
    # predict y based on x for the test data
    y_pred = classification_algorithm.predict(X_test)

    # store accuracy for this simulation
    mlp2_acc[loop_number] = accuracy_score(y_test, y_pred) * 100
    
print('{0:.2f} % accuracy with {1:0f} and {2:0f} hidden units in hidden layer 1 and 2, respectively.'.format(np.mean(mlp2_acc),
    layer1_hidden_units, layer2_hidden_units))
