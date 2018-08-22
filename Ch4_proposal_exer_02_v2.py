import delta_learning
import numpy as np

np.set_printoptions(suppress=True)

"""
To have an idea about the functions that are at your disposal, simply use
help(name_of_module)
"""
# help(delta_learning)

inputted_pattern = [1, -1, 1, -1, 1, 1, -1, -1]
desired_output = [1, -1, 1, -1, 1, 1]

altered_weight_matrix = delta_learning.loop_delta(inputted_pattern, desired_output, loops=500, print_loops=False)
