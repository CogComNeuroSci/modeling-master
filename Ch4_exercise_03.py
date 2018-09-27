#!/usr/bin/python3

import Ch0_delta_learning as delta_learning
import numpy as np

"""
@author: Pieter Huycke
email: pieter.huycke@ugent.be
"""

np.set_printoptions(suppress=True)

# ----------------------
# learning about Bocelli
# ----------------------

sight_bocelli = [.99, .01, .99, .01, .99, .01]
hearing_bocelli = [.99, .99, .01, .01]

# starting weights
one_weight = [.2] * len(sight_bocelli)
original_weights_bocelli = [one_weight] * len(hearing_bocelli)
altered_weights_bocelli = np.copy(original_weights_bocelli)

# delta learning Bocelli
print('Altering weight matrix for Bocelli...')
for i in range(10000):
    new_weights = delta_learning.weight_change(1.5, sight_bocelli, hearing_bocelli, altered_weights_bocelli,
                                               function_word='logistic')
    altered_weights_bocelli = new_weights

# ------------------------
# learning about Metallica
# ------------------------

sight_metallica = [.99, .99, .01, .01, .01, .99]
hearing_metallica = [.01, .99, .01, .99]

# starting weights
one_weight = [.2] * len(sight_metallica)
original_weights_metallica = [one_weight] * len(hearing_metallica)
altered_weights_metallica = np.copy(original_weights_metallica)

# delta learning Metallica
print('Altering weight matrix for Metallica...')
for j in range(10000):
    new_weights = delta_learning.weight_change(1.5, sight_metallica, hearing_metallica, altered_weights_metallica,
                                               function_word='logistic')
    altered_weights_metallica = new_weights

# --------
# Summary
# --------

print('\nAltered weight matrix for Bocelli:')
[print(weights) for weights in altered_weights_bocelli]

print('\nAltered weight matrix for Metallica:')
[print(weights) for weights in altered_weights_metallica]

# added weight matrices
print('\nAdded weight matrices:')
added_weights = np.add(altered_weights_bocelli, altered_weights_metallica)
[print(weights) for weights in added_weights]

# --------------------
# testing interference
# --------------------

seeing_bocelli = delta_learning.internal_input(sight_bocelli, added_weights, act_function='logistic')[0]
seeing_metallica = delta_learning.internal_input(sight_metallica, added_weights, act_function='logistic')[0]

print('\nSeeing Bocelli after learning two different patterns:')
print(np.round(seeing_bocelli, 2))
print('Desired output:')
print(hearing_bocelli)

print('\nSeeing Metallica after learning two different patterns:')
print(np.round(seeing_metallica, 2))
print('Desired output:')
print(hearing_metallica)
