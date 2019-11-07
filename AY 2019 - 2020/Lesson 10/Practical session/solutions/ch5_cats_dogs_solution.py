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

import matplotlib.pyplot as plt
import numpy             as np
import os
import pickle
import sklearn.datasets

import cv2

#%%

# --------- #
# CONSTANTS #
# --------- #

SAMPLESIZE   = 250
DIM_SIZE     = 80
SANITY_CHECK = False

ROOT         = r'C:\Users\pieter\Downloads\dogs-vs-cats'
TRAIN_DIR    = os.path.join(ROOT, 'train')
TEST_DIR     = os.path.join(ROOT, 'test')

#%%

# -------- #
# FUNCTION #
# -------- #

def process_image(file_location, file_name, dimension):
    
    '''
    :param str file_location: The path where the image is located
    :param str file_name: The name of the image
    
    :return: The rescaled (80 x 80) and grayscaled NumPy array depicting image
    '''
    
    file = os.path.join(file_location, file_name)
    imag = cv2.imread(file,
                      cv2.IMREAD_GRAYSCALE)
    return cv2.resize(imag,
                      dsize = (dimension, dimension))

#%%

# ------ #
# IMAGES #
# ------ #
    
# define our list of training and test samples
images     = os.listdir(TRAIN_DIR)
train_cats = images[:SAMPLESIZE]
train_dogs = images[-SAMPLESIZE:]

del images

if SANITY_CHECK:
    # show we are actually working with images in RGB values using a random image
    img = process_image(TRAIN_DIR, 
                        train_dogs[np.random.randint(100)], 
                        DIM_SIZE)
    plt.imshow(img,
               cmap = "gray")
    
    del img

#%%

# ------------------ #
# SAVE PROCESSED ARR #
# ------------------ #

# create arrays for both animals
data_cat = np.zeros((SAMPLESIZE, DIM_SIZE, DIM_SIZE))
data_dog = np.zeros((SAMPLESIZE, DIM_SIZE, DIM_SIZE))

for i in range(len(train_cats)):

    # cats
    img_cat = process_image(TRAIN_DIR, 
                            train_cats[i], 
                            DIM_SIZE)
    data_cat[i,:,:] = img_cat
    
    # dogs
    img_dog = process_image(TRAIN_DIR, 
                            train_dogs[i], 
                            DIM_SIZE)
    data_dog[i,:,:] = img_dog
    
del i, img_cat, img_dog, train_cats, train_dogs

#%%
 
# ------------ #
# SAVE TARGETS #
# ------------ #
   
# define the target arrays
cat_target = np.zeros(SAMPLESIZE)
dog_target = np.ones(SAMPLESIZE)

targets    = np.ravel(np.array((cat_target, dog_target)))

del cat_target, dog_target

#%%

# ----------- #
# FINAL ARRAY #
# ----------- #

# store together
cat_dog = np.vstack((data_cat, data_dog))

# shuffle together
indx = np.arange(len(targets))
np.random.shuffle(indx)

cat_dog = cat_dog[indx, :, :]
targets = targets[indx]

del indx

#%%

# ------ #
# CHECKS #
# ------ #

if SANITY_CHECK:
    # hopefully a cat
    plt.subplot(1, 2, 1)
    plt.imshow(data_cat[np.random.randint(len(data_cat))],
               cmap = "gray")
    plt.title('Cat?')
    
    # hopefully a dog
    plt.subplot(1, 2, 2)
    plt.imshow(data_dog[np.random.randint(len(data_dog))],
               cmap = "gray")
    plt.title('Dog?')

del data_cat, data_dog

#%%

# -------------- #
# WRITE TO BUNCH #
# -------------- #

description = 'The Asirra dataset (Animal Species Recognition for '         + \
              'Restricting Access) is contains millions of images of both ' + \
              'cats and dogs. The smaller dataset that you just loaded in ' + \
              'contains 500 pictures: 250 images of cats and 250 images '   + \
              'of dogs. These images are already processed for you to use ' + \
              'in your classifcation endeavors. Preprocessing means that '  + \
              'the images are grayscaled and all have the same dimensions.'
              
dataset = sklearn.datasets.base.Bunch(DESC         = description,
                                      images       = cat_dog, 
                                      target       = targets, 
                                      target_names = {0: 'cat', 
                                                      1: 'dog'})

del description, cat_dog, targets

f = open(os.path.join(ROOT, 'processed', 'cats_dogs.pkl'),"wb")
pickle.dump(dataset, f)
f.close()