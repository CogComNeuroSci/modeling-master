import delta_learning
import numpy as np

"""
To have an idea about the functions that are at your disposal, simply use
help(name_of_module)
"""
# help(delta_learning)

sight_peach = [1, -1, 1, -1, 1, 1, -1, -1]
name_peach = [1, 1, -1, -1]
sublist = [0, 0, 0, 0, 0, 0, 0, 0]
weights = [sublist] * 4

tester = np.array(delta_learning.internal_input(sight_peach, weights)[0])
new_matrix = delta_learning.weight_change(1, sight_peach, name_peach, weights,
                                          delta_learning.internal_input(sight_peach, weights))
print(new_matrix)

print(delta_learning.weight_change(1, sight_peach, name_peach, new_matrix,
                                   delta_learning.internal_input(sight_peach, new_matrix)))

print(delta_learning.activation_function(np.sum(np.multiply(sight_peach,
                                                            [-0.237, 0.237, -0.237, 0.237, -0.237, -0.237, 0.237, 0.237]))))