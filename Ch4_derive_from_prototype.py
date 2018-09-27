#!/usr/bin/python3

import numpy as np
import random
import os

"""
@author: Pieter Huycke
email: pieter.huycke@ugent.be
"""

os.chdir(r'C:\Users\Pieter\Downloads\Modeling\code\modeling-master')

# create temporary file
file = open("temp_patterns.txt", "w")
prototypical_pattern = np.array([.99, .01, .99, .99, .01, .01, .01, .01, .99, .99, .99, .99, .99, .01, .01, .01])
altered_pattern = np.copy(prototypical_pattern)

for i in range(50):
    changed = np.random.randint(low=2, high=6, size=1)
    which_change = np.random.randint(low=0, high=11, size=len(prototypical_pattern))
    for j in range(changed[0]):
        random_index = np.random.randint(low=0, high=len(prototypical_pattern), size=1)
        altered_pattern[random_index] = 1 - altered_pattern[random_index]
    np.savetxt(file, altered_pattern, delimiter='  ', fmt='%1.3f', newline='\r\n')
    altered_pattern = prototypical_pattern

file.close()

# create output file
print('Using temp file...')
new_file = open("patterns_different_dogs.txt", "w")

data = np.loadtxt("temp_patterns.txt").reshape(50, 16)
np.savetxt(new_file, data, delimiter='  ', fmt='%1.3f', newline='\r\n')

new_file.close()
print('Simulation complete\nRemoving temp file..')

os.remove("temp_patterns.txt")
print('All done')
