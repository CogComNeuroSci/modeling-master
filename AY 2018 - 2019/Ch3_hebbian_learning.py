#!/usr/bin/env python3

# Problem setting
"""
This code can be used to illustrate the idea of Hebbian learning
Depending on the learning rate and the activation of the linked neurons, the weight will change each cycle
When one of the neurons is not active, the weight will not change
"""

import matplotlib.pyplot as plt

# --------------------- #

cycles = int(input('How many hebbian cycles? '))

# ask for user input
# activation
activations = []

for i in range(2):
    inserted_1 = float(input('Please define the activation of neuron %d: ' % (i + 1)))
    activations.append(str(inserted_1))

# ask for user input
# weights
weights = []

for j in range(1):
    inserted_2 = float(input('Please define the starting weight from neuron %d to neuron %d: ' % (1, 2)))
    weights.append(str(inserted_2))

# ask for user input
# learning rate
beta = float(input("Define the learning rate: "))

# --------------------- #

for k in range(cycles):

    circle1 = plt.Circle((.25, .50), 0.05, color='black')
    circle2 = plt.Circle((.75, .50), 0.05, color='black')

    fig, ax = plt.subplots()

    ax.add_artist(circle1)
    ax.add_artist(circle2)

    line1 = plt.plot([.25, .75], [.50, .50], 'k-')

    plt.axis([0, 1, 0, 1])

    # text with the left neuron
    plt.text(.25, 0.58, activations[0], ha='center', va='center', transform=ax.transAxes)

    if k == 0:
        # text with the weights
        plt.text(.50, 0.56, weights[0], ha='center', va='center', transform=ax.transAxes)
        plt.text(.90, 0.90, 'Δw = 0', ha='center', va='center', transform=ax.transAxes)
        plt.text(.10, 0.90, 'cycle = 1', ha='center', va='center', transform=ax.transAxes)
    else:
        weights = list(map(float, weights))
        activations = list(map(float, activations))

        change = beta * activations[0] * activations[1]
        weights[0] = str(round(weights[0] + change, 4))
        plt.text(.50, 0.56, weights[0], ha='center', va='center', transform=ax.transAxes)
        weightchange = 'Δw = ' + str(round(change, 4))
        plt.text(.90, 0.90, weightchange, ha='center', va='center', transform=ax.transAxes)
        cycle_count = 'cycle = ' + str(k+1)
        plt.text(.10, 0.90, cycle_count, ha='center', va='center', transform=ax.transAxes)

    # text with the output node
    plt.text(.75, 0.58, activations[1], ha='center', va='center', transform=ax.transAxes)

    # visual stuff
    plt.title('Hebbian learning')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    while plt.waitforbuttonpress(0):
        plt.show()

    plt.close()
