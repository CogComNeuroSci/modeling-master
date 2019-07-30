# -*- coding: utf-8 -*-
"""
simple bayes calculator
Tom Verguts, July 2019
"""

# initialize
prior = 0.001
p11   = 0.99    # prob(ring/somebody there)
p10   = 0.01   # prob(ring/nobody there); not zero; I can e.g. hallucinate 

post = prior*p11/(prior*p11+(1-prior)*p10) # the posterior probability

print(post)