import delta_learning
import numpy as np

np.set_printoptions(suppress=True)

"""
To have an idea about the functions that are at your disposal, simply use
help(name_of_module)
"""
# help(delta_learning)

sight_peach = [1, -1, -1, 1, 1, 1]
name_peach = [1, 1, -1, -1]
sublist = [0, 0, 0, 0, 0, 0]
weights_1 = [sublist] * 4

weights = delta_learning.initialise_weights(sight_peach, name_peach, zeros=False)
print('Original weight matrix:\n', weights)

altered = delta_learning.weight_change(1, sight_peach, name_peach, weights)
print('Altered weight matrix:\n', altered)

for j in range(len(weights)):
    print('Original:', delta_learning.activation_function(np.sum(np.multiply(sight_peach, weights[j]))))
    print('Altered:', delta_learning.activation_function(np.sum(np.multiply(sight_peach, altered[j]))))