#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:58:13 2019

@author: tom verguts
plotting function for chapter 9 (RL-MDP) py files
variable n is binary: is it just a single plot (n=0) or not (n>0)
"""

def plot_value(fig, axs, row, column, value_matrix, title = "", n = 3, grid = True):
    offset = 0.05
    if n > 0:
        number = axs.shape[0]*row + column
        obj = axs.flat[number]
    else:
        obj = axs
    if len(value_matrix.shape)==2:        
        nrow = value_matrix.shape[0]
        ncol = value_matrix.shape[1]
    else:
        nrow = 1
        ncol = value_matrix.size
    obj.set_title(title)
    for xloop in range(nrow):
        for yloop in range(ncol):
            obj.text(offset + yloop/ncol, offset + ((nrow-1) - xloop)/nrow, "{:.1f}".format(value_matrix.flat[xloop*ncol + yloop]))
    obj.set_xticklabels([])
    obj.set_yticklabels([])
    obj.grid(grid)
    return