#!/usr/bin/env python3

# Problem setting
"""
author = pieter huycke (??)
This code can be used to illustrate the idea that output changes based on the values of the input and the weights of
the connection from the input unit to the output unit
In this toy network, we have three inputs, and thus three connections with specific weights from the input units to
the output unit
"""

import matplotlib.pyplot as plt

circle1 = plt.Circle((.25, .75), 0.05, color='black')
circle2 = plt.Circle((.25, .50), 0.05, color='black')
circle3 = plt.Circle((.25, .25), 0.05, color='black')

circle4 = plt.Circle((.75, .50), 0.05, color='black')

fig, ax = plt.subplots()

ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)

line1 = plt.plot([.25, .75], [.75, .50], 'k-')
line2 = plt.plot([.25, .75], [.50, .50], 'k-')
line3 = plt.plot([.25, .75], [.25, .50], 'k-')

plt.axis([0, 1, 0, 1])

# ask for user input
# activation

activations = []

for i in range(3):
    inserted_1 = float(input('Please define the activation of node %d: ' % (i+1)))
    activations.append(str(inserted_1))

# ask for user input
# weights

weights = []

for j in range(3):
    inserted_2 = float(input('Please define the weights from node %i to the output node: ' % (j+1)))
    weights.append(str(inserted_2))

# text with the input nodes
plt.text(.25, 0.85, activations[0], ha='center', va='center', transform=ax.transAxes)
plt.text(.25, 0.60, activations[1], ha='center', va='center', transform=ax.transAxes)
plt.text(.25, 0.35, activations[2], ha='center', va='center', transform=ax.transAxes)

# text with the weights
plt.text(.50, 0.67, weights[0], ha='center', va='center', transform=ax.transAxes)
plt.text(.50, 0.53, weights[1], ha='center', va='center', transform=ax.transAxes)
plt.text(.50, 0.42, weights[2], ha='center', va='center', transform=ax.transAxes)

# calculate output
for k in range(len(weights)):
    weights[k] = float(weights[k])
    activations[k] = float(activations[k])

output = 0

for l in range(len(weights)):
    output += (weights[l] * activations[l])

output = round(output, 4)
output = str(output)

# text with the output node
plt.text(.75, 0.58, output, ha='center', va='center', transform=ax.transAxes)

# visual stuff
plt.title('Toy network')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)

plt.show()
