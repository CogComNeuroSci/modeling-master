#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:11:55 2020

@author: tom
"""

import tensorflow as tf

a = tf.constant(3)
b = tf.constant(2)
c = tf.multiply(a, b)
d = tf.multiply(4, 2)
e = tf.multiply(5, 3)
f = tf.multiply(d, e)
g = tf.add(c, f)

with tf.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter("output", sess.graph)
    print(sess.run(g))
    writer.close()