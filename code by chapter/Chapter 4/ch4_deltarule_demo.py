#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:47:59 2018

@author: tom verguts
demo of the delta learning rule
with logistic activation function and cross-entropy error
"""
import numpy as np
import numpy.matlib as ml
import random
import matplotlib.pyplot as pl

def logist(x):
    return 1/(1+np.exp(-x))

def error(w):
    er = 0
    predictions = logist(np.dot(np.concatenate((x1, x2), axis = 0),w.T))
    for loop in range(2*n_patterns):
        er += ((loop<=(n_patterns-1)) - predictions[loop])**2
    er /= (2.*n_patterns)    
    return er

timesleep = 0.01
beta = 0.2
xmin, xmax, ymin, ymax = -10, 10, -10, 10
n_trials, n_patterns = 100, 5
mu1, mu2, s1, s2 = np.array([-2, -1]), np.array([1, 1]), 2, 1
xrange = np.linspace(xmin, xmax)
x1 = s1*np.random.randn(n_patterns, 2) + ml.repmat(mu1,n_patterns,1) # 'cats'
x1 = np.concatenate((x1, np.ones((n_patterns,1))), axis = 1)
x2 = s2*np.random.randn(n_patterns, 2) + ml.repmat(mu2,n_patterns,1) # 'dogs'
x2 = np.concatenate((x2, np.ones((n_patterns,1))), axis = 1)
w = np.random.randn(3)

fig, ax = pl.subplots()

for trial in np.arange(n_trials):
    pl.cla()
    ax.scatter(x1[:,0], x1[:,1], color = "green")
    ax.scatter(x2[:,0], x2[:,1], color = "red")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    sample = random.choice(range(2*n_patterns)) # a randomly sampled dog or cat
    target = sample<=(n_patterns-1) # cat = 1, dog = 0
    if target==1:
        x = x1[sample,:]
    else:
        x = x2[sample-n_patterns,:]
    prediction = logist(np.dot(w,x))    
    prediction_error = target-prediction
    delta = beta*x*prediction_error
    w += delta
    ax.plot(xrange, -w[0]/w[1]*xrange - w[2]/w[1], color = "black")
    fig.canvas.draw()
    pl.show()
    fig.waitforbuttonpress(0.1)
ax.set_title("end of optimization\n final error = {:.3}".format(error(w)))