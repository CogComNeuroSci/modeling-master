#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
GitHub:  phuycke
"""


#%%

# import: general and scikit-learn specific
import matplotlib.pyplot as plt
import numpy             as np
import os
import pandas            as pd
import seaborn           as sns

from sklearn.datasets import make_circles

#%%

# general plotting parameters
sns.set(style       = "ticks", 
        color_codes = True)

#%%

# make the dataset
coord, label = make_circles(n_samples    = 150,
                            noise        = .075, 
                            factor       = 0.5, 
                            random_state = 1)
coord += 4

# make it a Pandas DataFrame
df = pd.DataFrame({'Feature 1': coord[:,0], 
                   'Feature 2': coord[:,1],
                   'Class': label})

# remap the labels
mymap = {0: 'Flower A', 1: 'Flower B'}
df    = df.applymap(lambda s: mymap.get(s) if s in mymap else s)

#%%

# plot the data
ax = sns.scatterplot(x       = df['Feature 1'], 
                     y       = df['Feature 2'], 
                     hue     = df['Class'],
                     markers = ["o", "s"],
                     palette = sns.color_palette('colorblind', 2))

# Put the legend out of the figure
plt.legend(bbox_to_anchor = (1.05, 1), 
           loc            = 2, 
           borderaxespad  = 0.)

#%%

# reshape label, and put together
label   = np.reshape(label, (150, 1))
dataset = np.concatenate((coord, label),axis=1)

del coord, label

#%%

# save the data in a NumPy array
path = r'C:\Users\pieter\Downloads\GitHub\modeling-master\AY 2019 - 2020\Lesson 10\Practical session\exercises'
np.save(os.path.join(path, 'dataset_novice.npy'),
        dataset)