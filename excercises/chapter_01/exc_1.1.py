#!/usr/bin/env python3

# Problem setting
"""
Implement an algorithm that minimizes f(x) = A*(x-B)**2 + C.
Try out different values of A, B, and C.
What is their role in the optimization?
Try to reason it through first, then implement it.
"""


limit = 100

x = 2
B = C = 1

minimal = 7
a_min = 7

maximal = 7
a_max = 7


for a in range(0, limit + 1):
    A = a
    solution = A*(x-B)**2 + C
    if solution < minimal:
        minimal = solution
        a_min = A
    if solution > maximal:
        maximal = solution
        a_max = A

print("Keeping the values of x at", x, ", and the values of B and C both at", B, ":")
print("The equation has the lowest value (%d), when A equals %d" % (minimal, a_min))
print("The equation has the highest value (%d), when A equals %d\n" % (maximal, a_max))


x = 2
A = C = 1

minimal = 7
b_min = 7

maximal = 7
b_max = 7


for b in range(0, limit + 1):
    B = b
    solution = A*(x-B)**2 + C
    if solution < minimal:
        minimal = solution
        b_min = B
    if solution > maximal:
        maximal = solution
        b_max = B

print("Keeping the values of x at", x, ", and the values of A and C both at", A, ":")
print("The equation has the lowest value (%d), when B equals %d" % (minimal, b_min))
print("The equation has the highest value (%d), when B equals %d\n" % (maximal, b_max))

x = 2
A = B = 1

minimal = 7
c_min = 7

maximal = 7
c_max = 7


for c in range(0, limit + 1):
    C = c
    solution = A*(x-B)**2 + C
    if solution < minimal:
        minimal = solution
        c_min = C
    if solution > maximal:
        maximal = solution
        c_max = C

print("Keeping the values of x at", x, ", and the values of A and B both at", A, ":")
print("The equation has the lowest value (%d), when C equals %d" % (minimal, c_min))
print("The equation has the highest value (%d), when C equals %d\n" % (maximal, c_max))

x = 1

highestAll = 7
highestCombination = [0, 0, 0]

lowestAll = 7
lowestCombination = [0, 0, 0]

for a in range(0, limit + 1):
    for b in range(0, limit + 1):
        for c in range(0, limit + 1):
            A = a
            B = b
            C = c
            solution = A * (x - B) ** 2 + C
            if solution > highestAll:
                highestAll = solution
                highestCombination = [a, b, c]
            if solution < lowestAll:
                lowestAll = solution
                lowestCombination = [a, b, c]

print("In the case where x equals", x, "and all the other variables are ranging from", 0, "to", limit,":")
print("The smallest value is", lowestAll,
      "which can be obtained by selecting the following parameters:", lowestCombination)
print("The largest value is", highestAll,
      "which can be obtained by selecting the following parameters:", highestCombination)
