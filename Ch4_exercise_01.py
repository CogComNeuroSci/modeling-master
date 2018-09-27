#!/usr/bin/python3

import math
import matplotlib.pyplot as plt
import numpy as np


"""
@author: Pieter Huycke
email: pieter.huycke@ugent.be
"""


def logistic_activation(beta, netinput, theta):

    """
    :param beta:
    The 'beta' parameter represents the slope of the activation function.
    In this example, the larger this parameter value, the steeper the logistic function will be.

    :param netinput:
    The netinput for a certain input unit.
    This is usually calculated using the linearity principle (see ch04 p01).

    :param theta:
    The threshold value.
    If the netinput is larger than this threshold, the activation value will be located in the right part of the plot.

    :return:
    The value for a certain netinput when using a logistic activation function.
    This value is basically a transformation of the netinput.
    Doing this makes sure that input at a certain input node is always restrained between certain boundaries.
    """

    return 1 / (1 + math.exp(-beta*(netinput-theta)))


def maximize_screen():

    """
    :return:
    Maximizes the screen for pyplot.
    Different ways are used depending on the backend that is installed and used.
    """

    import matplotlib
    used_backend = matplotlib.get_backend()
    if used_backend == 'wxAgg':
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif used_backend == 'Qt4Agg':
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
    else:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

    return 'Used backend: %s' % used_backend


def plot_logistic_activation(beta, x_range, theta):

    """"
    :param beta:
    The 'beta' parameter represents the slope of the activation function.

    :param x_range:
    Defines the start and end value of the x-axis.
    The x-axis runs from -x_range to x_range.

    :param theta:
    The threshold value.

    :return:
    Here, we plot an activation function, calculated using three user-defined parameter values: beta, x_range, and theta.
    The first and the last parameter value are not changed within the function.
    For the netinput (which is necessary for calculating an activation value), we use all values within [-x_range, x_range], with increments of one.
    The results for each value of x_range are plotted.
    This plot represents a possible activation function, each different depending on the inputted parameter values.
    """

    if (x_range <= 1) or (x_range > 100):
        print('Predefined error'
              '\nFor visualisation purposes, we have restricted the "x_range" parameter.'
              '\nIn particular, x_range can only take values from the following interval: '
              '\n[2, 100]')

    x_axis = []
    y_axis = []
    for inputs in range(-x_range, x_range + 1):
        x_axis.append(inputs)
        y_axis.append(logistic_activation(beta, inputs, theta))

    plt.plot(x_axis, y_axis, 'k')
    plt.xlabel(r'$in_i$')
    plt.ylabel(r'$y_i \ (activation \ level \ for \ a \ certain \ in_i)$')
    plt.suptitle(r'$y_i:\frac{1}{1 + \exp(-\beta \times (in_i - \theta))}$')
    plt.title(r'$\beta: %1.2f \bullet in_i: [-%1.2f, %1.2f] \bullet \theta: %1.2f$' % (beta, x_range, x_range, theta), fontsize=10)
    plt.axis([-x_range, x_range, -.05, 1.05])
    if x_range <= 30:
        plt.xticks(np.arange(-x_range, x_range + .1, step=1))
    else:
        plt.xticks(np.arange(-x_range, x_range + .1, step=5))
    plt.yticks(np.arange(0, 1 + .1, step=.1))
    line1 = plt.hlines(.5, -x_range, x_range, colors='r')
    line2 = plt.axvline(x=theta)
    plt.legend((line1, line2), (r'$y_i \ (activation \ level) = .5$', r'$\theta \ (threshold \ value)$'))
    maximize_screen()
    plt.show()


def asking_input():

    """
    :return:
    Asks the user for input, and uses this to draw an activation function based on the provided parameter values.
    """

    print('Please define a value for the following parameters:\n'
          '\t\t1) beta (slope parameter)\n'
          '\t\t2) the range for the provided input [2, 100]\n'
          '\t\t3) theta (threshold value)\n\n'
          'An example of a possible input:\n'
          '.6 20 9\n\n'
          'It is clear that in this example:\n'
          '\t\t1) .6 represents a possible value for beta\n'
          '\t\t2) the x-axis will go from -20 all the way to 20\n'
          '\t\t3) 9 represents a possible value for theta\n'
          'They are separated by one space only.\n')
    inputted_vals = input("Enter your desired parameter values: ")
    if not inputted_vals:
        print('No input provided, standard values assumed.\n')
        beta_var, x_var, theta_var = .6, 20, 9
        plot_logistic_activation(beta_var, x_var, theta_var)
    elif len(inputted_vals.split()) != 3:
        print('\nPlease input three parameter values only separated by a space\nRestarting program...\n')
        main()
    else:
        beta_var, x_var, theta_var = inputted_vals.split()
        beta_var, theta_var = float(beta_var), float(theta_var)
        x_var = int(x_var)
        plot_logistic_activation(beta_var, x_var, theta_var)


def main():

    """
    :return:
    Starts the main program and asks for user input.
    """

    print(' -------------- \n'
          '| General idea |\n'
          ' -------------- \n'
          'This first exercise is an easy one to get in touch with the course material.\n'
          'In the beginning of Chapter 04 (the Delta rule), we have learned that an activation function is \n'
          'used to transform the "netinput" of a unit (calculated '
          'using the linearity principle) to an "activation level".\n\n'
          'Different types of activation functions are available: linear, logistic, threshold, sigmoid... .\n'
          'In the course, we mainly focused on the logistic activation function.\n'
          'We noted that several parameter values impact how exactly the netinput is transformed to an activation '
          'value.\n\n'
          'In this exercise, you are going to use this program to manipulate these parameter values, '
          '\nand investigate how these manipulations impact the resulting activation function.\n\n'
          'Specifically, you should study how the activation function changes by altering the:\n'
          '\t\t1) beta parameter (the slope)\n'
          '\t\t2) theta parameter (the threshold)\n')

    while True:
        print(' ------------------- \n'
              '| Kicking the tires |\n'
              ' ------------------- ')
        yes_or_no = str(input('Would you like to draw an activation function? (y/n): '))
        if yes_or_no.lower() in ['yes', 'y']:
            asking_input()
        elif yes_or_no.lower() in ['no', 'n']:
            print('Program terminating...')
            break
        else:
            print('Provide correct input.')
            pass


main()
