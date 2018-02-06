#!/usr/bin/env python3

# Problem setting
"""
Implement an algorithm that minimizes f(x) = A*(x-B)**2 + C.
Try out different values of A, B, and C.
What is their role in the optimization?
Try to reason it through first, then implement it.
"""

from sympy import symbols
from sympy import plot


def main():

    """
    Plots a function based on a considered formula
    The three values A, B and C are inputted by the user
    x is varied in a range specified by the user
    """

    formula = str(input("Please input the formula you want to plot: "))

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

    new_formula = formula.lower()
    letters.remove(variable)
    looped = 0

    while looped != len(letters):
        letter_replaced = chr(96 + (looped + 1))
        value = str(input("What value do you specify for parameter %s: " % letter_replaced.upper()))
        new_formula = new_formula.replace(letter_replaced, value)
        looped += 1

    correct = False

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

    plot(new_formula, (var, range_var_low, range_var_high),
         title="Function created based on inputted values",
         ylabel='Function values',
         xlabel='Value for x')


main()
