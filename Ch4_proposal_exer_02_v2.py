import delta_learning
import numpy as np

np.set_printoptions(suppress=True)

"""
To have an idea about the functions that are at your disposal, simply use
help(name_of_module)
"""
# help(delta_learning)

inputted_pattern = [1, -1, 1, -1, 1, 1, 1, -1]
desired_output = [1, -1, 1, -1, 1, 1]

weight_zero = [0]*len(inputted_pattern)
weight_matrix = [weight_zero] * len(desired_output)

print(weight_matrix)
print(delta_learning.activation_function(0, form='linear'))

