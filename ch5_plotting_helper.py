#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 

Helper function used in exercise 02 for chapter 05
"""

# import: general and scikit-learn specific
import itertools
import numpy                 as np
import matplotlib.pyplot     as plt


def plot_confusion_matrix(cm, 
                          classes,
                          normalize  = False, 
                          plot_title = 'Confusion matrix', 
                          colorcode  = plt.cm.Blues,
                          show       = True):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
    else:
        print(cm)

    if not show:
        print(cm)
        print('For a visualisation, set the argument "show" to "True"...')
        return 0
 
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap = colorcode)
    plt.title(plot_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Correct answer')
    plt.xlabel('Answer given by model')

    plt.tight_layout()
    plt.show()
    plt.waitforbuttonpress(0)
    plt.close()
