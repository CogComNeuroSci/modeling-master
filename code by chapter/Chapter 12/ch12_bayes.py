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


#%%prior considerations

import matplotlib.pyplot as plt
import numpy as np

D = (6, 3) # the data!
alpha, beta = 1, 1

x = np.linspace(0, 1, num = 50)
y = np.multiply( x**(alpha-1+D[0]), (1-x)**(beta-1+D[1]) )
y_trunc = np.multiply(1*(x>0.5), y)
y = y/np.sum(y)
y_trunc = y_trunc/np.sum(y_trunc)

# plot
fig, axes = plt.subplots(nrows = 1, ncols = 1)
axes.plot(x,y, color = "black", linestyle = "dashed", label = "no prior info")
axes.plot(x, y_trunc, color = "black", label = "with prior info")
axes.legend()
plt.xlabel("p")
plt.ylabel("Posterior(p)")

#%% loss functions

import matplotlib.pyplot as plt
import numpy as np

g       = np.linspace(-3, +3, num = 50)
support = np.linspace(-3, +3, num = 50)
ds      = support[1] - support[0]
loss    = np.linspace(-3, +3, num = 50)
mean, sigma   = 0, 1
con     = 1/(sigma*np.sqrt(2*np.pi))


for idx, g_loop in enumerate(g):
    som = 0
    for s in support:
        som += np.exp(-1/(2*sigma)*(s-mean)**2)*np.abs(g_loop-s)
    loss[idx] = som*ds*con    

plt.plot(g, loss)