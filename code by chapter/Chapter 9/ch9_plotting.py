#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:58:13 2019

@author: tom verguts
plotting functions for chapter 9 (RL-MDP) py files
"""
import numpy as np

def plot_value(fig, axs, row, column, value_matrix, title = "", n = 3, grid = True):
    """for plotting values in linear or rectangular worlds (lineworld, gridworld)
    variable n is binary: is it just a single plot (n=0) or not (n>0)"""
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

def plot_value_circular(fig, axs, plot_loop, value, r, centerx, centery, length, resolution, states):
    """for plotting values in a circular world (ringworld)"""
    for loop in range(resolution):
        angle = 1-loop/resolution*2*np.pi
        axs[plot_loop].scatter( centerx + length*np.cos(angle), centery + length*np.sin(angle), s = 0.1, c = [[0.1, 0.01, 0.01]] )                
    for loop in range(len(r)):
        angle = (1-loop/len(r))*2*np.pi
        axs[plot_loop].set_axis_off()
        if plot_loop == 0:
            to_plot = states[loop]
        elif plot_loop == 1:
             to_plot = "{}".format(r[loop])
        else:
            to_plot = "{:.2f} // {:.2f}".format(value[loop, 0], value[loop, 1])
        fig.subplots_adjust(hspace=0.7)
        axs[plot_loop].text(centerx + length*np.cos(angle), centery + length*np.sin(angle), to_plot)