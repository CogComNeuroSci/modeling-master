#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""


#%%

# ---------- #
# LOAD NUMPY #
# ---------- #

import numpy as np

#%%

# ----------------------------------- #
# QUESTION 1: slicing + adding arrays #
# ----------------------------------- #

"""
Below you find the Python code for creating two NumPy arrays.
Your job is to indicate what would happen if we ran this block of code.
"""

arr_1  = arr_2 = np.array([1, 2, 3, 4, 5])

summed = arr_1[:2] + arr_2[2, 4]
print(summed)

"""
Possible answers:
    a) A  ValueError is raised
    b) An IndexError is raised
    c) [4. 7.]
    d) [3. 6.]
    
--> b is correct, because you try to access row 3 and column 4 of arr_2
    This does not exist, as arr_2 is of size (1, 5)
"""


#%%

# -------------------------- #
# QUESTION 2: array creation #
# -------------------------- #

"""
Your goal is to create a matrix of data that has the following form

Out[1]:
array([[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0],
       [0, 0, 1],
       [0, 1, 0],
       [1, 0, 0],
       [0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]])

Your starting point is the NumPy array 'arr':

arr
Out[2]: 
array([[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]])
    
You are thinking of three NumPy functions that might be of use to you.
What functions are able to *directly* yield the aforementioned output?

a. np.tile()
b. np.vstack()
c. np.copy()
"""

arr     = np.array([[0, 0, 1], 
                    [0, 1, 0], 
                    [1, 0, 0]])

using_1 = np.tile(arr, (3, 1))
using_2 = np.repeat(arr, 3, axis = 0)
using_3 = None


""" 
Possible answers:
    a) a and b but not c
    b) a and c but not b
    c) b and c but not a
    d) a, b and c
    
--> a is correct, both np.tile() and np.vstack() are able to produce the 
    desired output. This is proven in the example code defined above.
"""

#%%

# ------------------------------- #
# QUESTION 3: understanding loops #
# ------------------------------- #

"""
Your task is to loop over an Array filled with n values.
From the moment you encounter the value v, you should stop looping and continue
with the rest of your script.

How would you handle this situation?
"""

"""
Possible answers:
I would:
    a) use a while loop
    b) use a for loop 
    c) be able to use either a while loop or a for loop
    d) search by hand until I find value v
    
--> c is correct, using a for loop with a break statement is also an option.
"""

#%%

# -------------------------- #
# QUESTION 4: function calls #
# -------------------------- #

"""
You have some custom-made trivial functions:
    - raising_power(integer, exponent)
    - make_array(scalar, length)
    
Below you use your functions in a custom-made piece of Python code

Following the code, what would be the output of the last command?
"""

# raise a number ('integer') to the power of 'exponent'
def raising_power(integer, exponent):
    return integer ** exponent

# transform a number into an array of size '1 x length' filled with 'scalar'
def make_array(scalar, length):
    arr = np.array([scalar] * length)
    return arr

var_1 = make_array(3, 5)
res_1 = raising_power(var_1, 2)

var_2 = raising_power(3, 2)
res_2 = make_array(var_2, 5)

res_1 == res_2

"""
Possible answers:
    a) array([ True,  True,  True,  True,  True])
    b) True
    c) array([ False,  False,  False,  False,  False])
    d) False
    
--> a is correct, NumPy does elementwise comparison of the array elements.
    Raising an array to the power n is the same as making an array of a scalar
    that was previously raised to the power of n.
"""

#%%

# --------------------------------- #
# QUESTION 5: reading control flows #
# --------------------------------- #

"""
You receive a code snippet from another Python enthusiast who ran into some
trouble when trying to define a while loop.
She asks you whether you can tell her when the while loop will stop running.

Can you help her out?
"""

# array filled with numbers (1 to 20, both included, then shuffled)
number_list = [10, 15,  2, 12,  7,  
                3, 17,  5,  4, 18, 
               20,  8, 14,  1, 11,
               13, 16,  6, 19,  9]

# needed variables
counter    = 0
list_index = 0

# checking conditions
while counter <= 4:
    counter    += int(number_list[list_index] / 10)
    list_index += 1

print("The value for list_index upon ending the loop equals: {}".format(list_index))

"""
Possible answers:
    a) 19
    b) 9
    c) The while loop will never stop running
    d) 10
    
--> d is correct, the while loop keeps going until counter is larger than 4. 
    Counter will increment with 1 if number_list[index] >= 10, so 
    when the fifth element is encountered that is larger than 10, the loop will
    break. This will happen with element '18', which is the tenth element.
"""

#%%

# ------------------------- #
# QUESTION 6: array slicing #
# ------------------------- #

"""
We consider the following NumPy array:
    
arr
Out[1]:
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10],
       [11, 12, 13, 14, 15],
       [16, 17, 18, 19, 20]])
    
Which of the following statements is entirely correct?

1. arr[1,:] equals [ 4, 9, 14, 19]
   arr[,:3] equals [ 6, 7,  8,  9, 10]
2. arr[1,:] equals [ 3, 8, 13, 18]
   arr[:,3] equals [ 1, 2,  3,  4,  5]
3. arr[1,:] equals [ 6, 7,  8,  9, 10]
   arr[,:3] equals [ 4, 9, 14, 19]
4. arr[1,:] equals [ 1, 2,  3,  4, 5]
   arr[,:3] equals [ 3, 8, 13, 18]
"""

# how to create the data that was used in this question
arr = np.arange(1, 21)
arr = np.reshape(arr, (4, 5))

"""
Possible answers:
    a) statement 1
    b) statement 2
    c) statement 3
    d) statement 4
    
--> c is correct, arr[1,:] is the second row of arr, and arr[,:3] equals
    its fourth column.
"""

#%%

# ----------------------------- #
# QUESTION 7: logical operators #
# ----------------------------- #

"""
Below, several logical statements are given.
Your job is to find out for which statements something is printed, and for 
which one, the printing is skipped.
"""

house = ['last1',  'last2',  'last3',  'last4',  'last5']
name  = ['first6', 'first2', 'first3', 'first4', 'first5']

# logical statement 1
if 'first6' in name and 'last6' not in house:
    pass
else:
    print('Printing 1')

# logical statement 2
if name[3] == 'first4' and house[3] == 'first3' or 'first2' in name:
    print('Printing 2')
else:
    pass

# logical statement 3
last_name = 'first' + str(5)
if last_name in name:
    for i in range(1, 6):
        string = 'first' + str(i)
        if string in name:
            pass
        elif 'first4' in name:
            break
        else:
            pass
        print('Statement 3')


# logical statement 4
boolean_sum = (False or True)       * \
              (False and False)     - \
              (not(False) and True) + \
              (True + (False or not(True)))
result      = int(boolean_sum)
if result > 0:
    print('Printing 4')
else:
    pass

"""
Possible answers:
    a) For statements 1 and 3
    b) For statements 2 and 4
    c) Only for statement 3
    d) None of the above
    
--> d is correct, as only something will be printed for statement 2.
"""