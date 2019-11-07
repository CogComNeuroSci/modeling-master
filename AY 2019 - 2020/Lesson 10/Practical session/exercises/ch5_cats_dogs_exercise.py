#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""


#%%

# -------------- #
# IMPORT MODULES #
# -------------- #

import numpy   as np
import os
import pickle

#%%

os.chdir(r'C:\Users\pieter\Downloads\dogs-vs-cats\processed')

objects = []
with (open("cats_dogs.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


dataset = objects[0]

del objects