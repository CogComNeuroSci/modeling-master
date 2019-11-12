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

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

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
    
    file = os.path.join(file_location, file_name)
    imag = cv2.imread(file,
                      cv2.IMREAD_GRAYSCALE)
    imag = cv2.resize(imag,
                      dsize = (dimension, dimension)) / 255.0
    return imag


#%%

# load in the data
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

# some background
print(dataset.DESC)

# load data
X = dataset.images
y = dataset.target

#%%

model = Sequential()

model.add(Conv2D(32, 
                 (3, 3), 
                 activation  = 'relu', 
                 input_shape = X.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 
                 (3, 3), 
                 activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, 
                 (3, 3), 
                 activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,
                activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, 
                activation = 'sigmoid'))

model.compile(loss      = 'binary_crossentropy', 
              optimizer = 'adam', 
              metrics   = ['accuracy'])


model.fit(X, 
          y, 
          epochs           = 15, 
          batch_size       = 15, 
          validation_split = 0.2)

#%%

# ---------------------- #
# CLASSIFY 10 NEW IMAGES #
# ---------------------- #

TEST_DATA   = r'C:\Users\pieter\Downloads\dogs-vs-cats\test'
test_images = os.listdir(TEST_DATA)

# make array to store the data
test_data  = np.zeros((10, 80, 80))
random_img = np.random.randint(0, len(test_images), 10)

for i in range(len(random_img)):
    
    img              = process_image(TEST_DATA,
                                     test_images[random_img[i]],
                                     80)
    test_data[i,:,:] = img

test_data = test_data.reshape(-1, 80, 80, 1)

predictions   = model.predict(test_data)
predicted_val = [int(round(p[0])) for p in predictions]

#%%

label_translation = {1: 'Dog',
                     0: 'Cat'}

# hopefully a cat
for i in range(len(test_data)):
    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    
    img = cv2.imread(os.path.join(TEST_DATA, 
                                  test_images[random_img[i]]))
    plt.imshow(img, 
               cmap          = plt.cm.gray_r, 
               interpolation = 'nearest')
    plt.title('Prediction: {}'.format(label_translation.get(predicted_val[i])))






