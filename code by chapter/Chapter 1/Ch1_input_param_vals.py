#!/usr/bin/env python3


"""
Algorithm to plot functions, such as f(x) = A*(x-B)**2 + C.
Try out different values of A, B, and C to get an intuition on their role in the function
Try to reason it through first, then implement it.
You may only see the plot(s) after termination of the program
Code by Pieter Huycke
"""

from sympy import symbols
from sympy import plot


def plotFunc():

    """
    Plots a function based on a considered formula
	In the example f(x) = A*(x-B)**2 + C,
    the three values A, B and C are inputted by the user;
    x is the variable, and it varies in a range specified by the user
    """

    satisfied = False
    while not satisfied:
        formula = str(input("Please input the formula you want to plot: "))
        print("Confirm formula?\n"
              "\tPress y to confirm.\n"
              "\tpress n to re-enter.")
        confirm = str(input("User input: "))
        if confirm.lower() in "yess":
            satisfied = True

    parsed = list(formula)
    letters = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    checked = 0
    while checked != len(parsed):
        if parsed[checked].lower() in alphabet and parsed[checked].lower() not in letters:
            letters.append(parsed[checked])
        checked += 1

    print("The following unique letters are detected in the formula:", letters)
    possible = "".join(letters)

    passing = False

    while not passing:
        variable = str(input("Which one of these letters is considered the variable? "))
        if variable not in possible:
            print('! ! !\n'
                  'Chosen value has to be available in', letters, '.\n'
                  '! ! !')
            continue
        else:
            passing = True

    new_formula = formula
    letters.remove(variable)
    print("Remaining parameters that need values:", letters)

    looped = 0

    while looped != len(letters):
        letter_replaced = letters[looped]
        value = str(input("What value do you specify for parameter %s: " % letters[looped]))
        new_formula = new_formula.replace(letter_replaced, value)
        looped += 1

    correct = False
    print("Formula with filled in values:", new_formula)

    while not correct:
        range_var_low = int(input("Please the LOWEST value for the variable (%s) in the function: " % variable))
        range_var_high = int(input("Please the HIGHEST value for the variable (%s) in the function: " % variable))
        if range_var_low >= range_var_high:
            print('! ! !\n'
                  'The lowerbound has to be lower than the upperbound.\n'
                  'Please input new values'
                  '! ! !')
            continue
        else:
            correct = True

    var = symbols(variable)

    print("... plotting ...")
    plot(new_formula, (var, range_var_low, range_var_high),
         title="Function created based on inputted values",
         ylabel='Function values',
         xlabel='Value for varying parameter')


def main():

    """
    Main function to get the program rolling
    """

    satisfied = False

    while not satisfied:
        print("Do you want to start the program?\n"
              "\tPress y to start program.\n"
              "\tpress n to terminate.")
        ask_input = str(input("User input: "))
        if ask_input.lower() in "yess":
            plotFunc()
        else:
            print("---------------------------\n"
                  "The program is now finished\n"
                  "---------------------------")
            satisfied = True


main()
