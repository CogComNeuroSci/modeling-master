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

import cv2
import matplotlib.pyplot as plt
import numpy             as np
import os
import pickle
import sklearn.datasets

from tqdm import tqdm

#%%

# --------- #
# CONSTANTS #
# --------- #

DIM_SIZE     = 80
SANITY_CHECK = True
CAT_COUNT    = 0
DOG_COUNT    = 0 

ROOT         = os.getcwd()
TRAIN_DIR    = os.path.join(ROOT, 'train')

#%%

# -------- #
# FUNCTION #
# -------- #

def process_image(file_location, file_name, dimension):
    
    '''
    :param str file_location: The path where the image is located
    :param str file_name: The name of the image
    
    :return: The rescaled (80 x 80), normalized and grayscaled 
             NumPy array depicting image
    '''
    
    file  = os.path.join(file_location, file_name)
    imag  = cv2.imread(file,
                       cv2.IMREAD_GRAYSCALE)
    imag  = cv2.resize(imag,
                       dsize = (dimension, dimension))
    
    imag  = imag.astype('float64')
    imag -= np.mean(imag)
    
    return imag

#%%

# ------ #
# IMAGES #
# ------ #
    
# define our list of training and test samples
images = os.listdir(TRAIN_DIR)

if SANITY_CHECK:
    # show we are actually working with images in RGB values using a random image
    img = process_image(TRAIN_DIR, 
                        images[np.random.randint(15)], 
                        DIM_SIZE)
    plt.imshow(img,
               cmap = "gray")
    
    del img

#%%

# ------------------ #
# SAVE PROCESSED ARR #
# ------------------ #

# count instances of each animal
for file_name in images:
    if 'dog' in file_name.lower():
        DOG_COUNT += 1
    elif 'cat' in file_name.lower():
        CAT_COUNT += 1
    else:
        raise Warning('Unknown file found in folder')

del file_name

# create data -, and target arrays for both animals
data    = np.zeros((CAT_COUNT + DOG_COUNT, DIM_SIZE, DIM_SIZE))
targets = np.zeros((CAT_COUNT + DOG_COUNT))

# loop over all files and process them (and show progress bar)
for i in tqdm(range(len(images))):

    # dog = 1 / cat = 0
    if 'cat' in images[i].lower():
        img_cat = process_image(TRAIN_DIR, 
                                images[i], 
                                DIM_SIZE)
        data[i,:,:] = img_cat
    else:
        targets[i] = 1
        img_dog = process_image(TRAIN_DIR, 
                                images[i], 
                                DIM_SIZE)
        data[i,:,:] = img_dog
    
del i, img_cat, img_dog, images

#%%

# ------ #
# CHECKS #
# ------ #

if SANITY_CHECK:
    # hopefully a cat
    plt.subplot(1, 2, 1)
    plt.imshow(data[np.random.randint(0, len(data) // 2)],
               cmap = "gray")
    plt.title('Cat?')
    
    # hopefully a dog
    plt.subplot(1, 2, 2)
    plt.imshow(data[np.random.randint(len(data) // 2, len(data))],
               cmap = "gray")
    plt.title('Dog?')

#%%

# ----------- #
# FINAL ARRAY #
# ----------- #

#  final manipulations before storage (normalize + reshape)
data  = np.array(data).reshape(-1, DIM_SIZE, DIM_SIZE, 1)

#%%

# -------------- #
# WRITE TO BUNCH #
# -------------- #

description = 'The small dataset that you are working with at this moment '   + \
              'is the result of scraping Google images for 15 pictures of '   + \
              'cats, and 15 images of dogs. Mind that preprocessing has '     + \
              'already been done. This means that the images are grayscaled ' + \
              ', and all have the same dimensions.'
              
dataset = sklearn.datasets.base.Bunch(DESC         = description,
                                      images       = data, 
                                      target       = targets, 
                                      target_names = {0: 'cat', 
                                                      1: 'dog'})

del description, data, targets

f = open(os.path.join(ROOT, 'processed', 'cats_dogs.pkl'),"wb")
pickle.dump(dataset, f)
f.close()