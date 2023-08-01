#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:25:50 2023

@author: tom verguts
attention
"""

import tensorflow as tf
from tf.keras.layers import MultiHeadAttention
from tf.keras import Input

layer = MultiHeadAttention(num_heads = 1, key_dim = 2)

target = Input(shape = [8, 16])
source = Input(shape = [4, 16])

output_tensor, weights = layer(target, source, return_attention_score = True)
print(output_tensor.shape)
