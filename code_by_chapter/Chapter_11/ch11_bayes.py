# -*- coding: utf-8 -*-
"""
simple bayes calculations
Tom Verguts, July 2019 (and later)
"""

#%% bayes posterior probability
# initialize
prior = 0.01
p11   = 0.99    # prob(ring/somebody there)
p10   = 0.01   # prob(ring/nobody there); not zero; I can e.g. hallucinate 

post = prior*p11/(prior*p11+(1-prior)*p10) # the posterior probability

print(post)

#%% exercise 11.2: the effect of more informative priors
# if you know (for sure) that x > 0.5, the posterior at the remaining (x > 0.5) points becomes sharper

import matplotlib.pyplot as plt
import numpy as np

D = (6, 3) # the data!
alpha, beta = 1, 1

x = np.linspace(0, 1, num = 50)
y = np.multiply( x**(alpha-1+D[0]), (1-x)**(beta-1+D[1]) )
y_trunc = np.multiply(1*(x>0.5), y) # truncated data: you are sure that x > 0.5 (eg, from prior data)
y = y/np.sum(y)
y_trunc = y_trunc/np.sum(y_trunc)   # normalize the truncated data: bcs np.sum(y_trunc) is smaller, the surviving data points "receive" more posterior value than in y

# plot
fig, axes = plt.subplots(nrows = 1, ncols = 1)
axes.plot(x,y, color = "black", linestyle = "dashed", label = "no prior info")
axes.plot(x, y_trunc, color = "black", label = "with prior info")
axes.legend()
axes.set_xlabel("p")
axes.set_ylabel("Posterior(p)")