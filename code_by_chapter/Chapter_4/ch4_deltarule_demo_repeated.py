#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:47:59 2018

@author: tom verguts
demo of the delta learning rule
with logistic activation function and cross-entropy error
this version does it repeatedly; first it categorises red vs green, then blue vs yellow
to demonstrate catastropic interference
"""

#%% imports
import numpy as np
import numpy.matlib as ml
import random
import matplotlib.pyplot as plt

#%% functions
def logist(x):
    return 1/(1+np.exp(-x))

def error(w):
    er = 0
    predictions = logist(np.dot(np.concatenate((x1[distribution_loop,:,:], x2[distribution_loop,:,:]), axis = 0),w.T))
    for loop in range(2*n_patterns):
        er += ((loop<=(n_patterns-1)) - predictions[loop])**2
    er /= (2.*n_patterns)    
    return er

#%% initialisations
fig, ax = plt.subplots()
n_distributions = 2
colors = ["red", "green", "blue", "yellow"]
timesleep = 0.01
beta = 0.1
xmin, xmax, ymin, ymax = -10, 10, -10, 10
n_trials, n_patterns = 100, 5
xrange = np.linspace(xmin, xmax)
x1 = np.zeros((n_distributions, n_patterns, 3))
x2 = np.zeros((n_distributions, n_patterns, 3))
hyper_mu1, hyper_mu2 = [0, 0], [0, 0] # hyper-parameters from which means are sampled
hyper_s = 3 # hyper-parameter standard deviation
s1, s2 = 0.5, 0.5 # noise standard deviations (first-order parameters)

#%% main code
for distribution_loop in range(n_distributions):
    mu1 = hyper_mu1 + np.random.randn(1,2)*hyper_s
    mu2 = hyper_mu2 + np.random.randn(1,2)*hyper_s

    x1[distribution_loop,:,:2] = s1*np.random.randn(n_patterns, 2) + ml.repmat(mu1,n_patterns,1) # 'cats'
    x1[distribution_loop,:,:]  = np.concatenate((x1[distribution_loop, :, :2], np.ones((n_patterns,1))), axis = 1)
    x2[distribution_loop,:,:2] = s2*np.random.randn(n_patterns, 2) + ml.repmat(mu2,n_patterns,1) # 'dogs'
    x2[distribution_loop,:,:]  = np.concatenate((x2[distribution_loop, :, :2], np.ones((n_patterns,1))), axis = 1)
    w = np.random.randn(3)

    for trial in np.arange(n_trials):
        plt.cla()
        for small_distribution_loop in range(distribution_loop+1):
            ax.scatter(x1[small_distribution_loop,:,0], x1[small_distribution_loop,:,1], color = colors[small_distribution_loop*2])
            ax.scatter(x2[small_distribution_loop,:,0], x2[small_distribution_loop,:,1], color = colors[small_distribution_loop*2+1])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        sample = random.choice(range(2*n_patterns)) # a randomly sampled dog or cat
        target = sample<=(n_patterns-1) # cat = 1, dog = 0
        if target==1:
            x = x1[distribution_loop, sample, :]
        else:
            x = x2[distribution_loop, sample-n_patterns, :]
        prediction = logist(np.dot(w, x))    
        prediction_error = target-prediction
        delta = beta*x*prediction_error
        w += delta
        ax.plot(xrange, -w[0]/w[1]*xrange - w[2]/w[1], color = "black")
        ax.plot([0, w[0]], [-w[2]/w[1], w[1] - w[2]/w[1]], color = "red")
        fig.canvas.draw()
        plt.show()
        plt.waitforbuttonpress(timesleep)
    ax.set_title("end of optimization\n final error = {:.3}".format(error(w)))