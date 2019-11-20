#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""


#%%

# import relevant modules
import os
import pickle

#%%

# load in the data
location = r'full\path\to\cats_dogs.pkl'
location = r'C:\Users\pieter\Downloads\GitHub\modeling-master\AY 2019 - 2020\Lesson 10\Practical session\solutions\downloads\processed'
current  = os.getcwd()

os.chdir(location)
objects = []
with (open("cats_dogs.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


dataset = objects[0]

del objects

os.chdir(current)

#%%

# some background on the data
print('')
print(dataset.DESC)

# keep the relevant part of your dataset, and organize them

#%%

# shuffle and train test split

#%%

# fit an MLP and check prediction accuracy
