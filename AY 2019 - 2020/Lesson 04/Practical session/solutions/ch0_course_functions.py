#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:15:54 2018

@author: Mehdi Senoussi

Todo:
    - many things..
    - ..
"""

import matplotlib.pyplot as pl
import numpy as np
import time


global pl, np, time, cols
cols = [[.8, .2, .1], [.2, .3, .9]]



def no_recurrence(weights):
    weights[np.identity(len(weights), dtype=np.bool)] = 0
    return weights


def plot_network(figsize = [13, 7], activations = np.random.rand(3),
                 weights = np.random.rand(3, 3), layers = np.array([1, 1, 2]),
                 energy = None):
   '''
    plot_network(figsize = [13, 7], activations = np.random.rand(3),
                 weights = np.random.rand(3, 3), layers = np.array([1, 1, 2]),
                 energy = None)

    Creates a figure with 3 subplots: (1) a network, (2) each weight for each
    cycle, (3) activations of each unit and the energy of the network if provided.
    
    This function returns the figure and axes, the texts and lines handles for
    the network and its unit positions to be used in the update_network function.
    
    Parameters
    ----------
    figsize : a list or array of the size of the figure window
    activations : array of N activations (floats or integers) for each of the
                  units in the network
    weights : array of N*N numbers (floats or integers) representing the
              connection weights for all units only half of the matrix is used,
              it's always weights[a, b] with a <= b
    layers : an array of integers representing which unit  belongs to which
             layer in the network, e.g. [1, 1, 2] for a 3 unit network with 2
             units on the first layer and one on the second layer.
    energy : (optional) the energy of the network which will be plotted on the
             bottom left subplot in red.
    
    Returns
    -------
    fig : handle of the figure
    axs : handle of the axes
    texts_handles : handle of the texts (activations and cycle number)
    lines_handles : handle of the lines (weights)
    unit_pos : the position of the units to re-draw weight lines
    
    '''
    fig = pl.figure(figsize = figsize)
    axs = []; circles = []
    axs.append(fig.add_subplot(2, 2, (1, 3)))
    axs.append(fig.add_subplot(2, 2, 2))
    axs.append(fig.add_subplot(2, 2, 4))

    n_layers = len(np.unique(layers))
    n_units = activations.shape[0]

    # computing where the
    distx = 1. / (n_layers + 1)
    disty = np.array([1. / (sum(layers == lay_n) + 1) for lay_n in np.unique(layers)])


    unit_pos = []
    for lay_n in np.unique(layers):
        n_unit_layer = sum(layers == lay_n)
        for i in range(n_unit_layer):
            unit_pos.append([lay_n * distx, (i+1) * disty[lay_n - 1]])
    unit_pos = np.array(unit_pos)

    # plot lines representing connections between units
    lines_handles = {}
    for lay_n in np.unique(layers):
        lay_units = np.where(layers == lay_n)[0]
        nextlay_units = np.where(layers == lay_n + 1)[0]
        for unit_n in lay_units:
            # plot lines between units of layer N
            for unit_n2 in lay_units:
                w = weights[unit_n, unit_n2]
                if w:
                    lines_handles['line_%i-%i' % (unit_n, unit_n2)] =\
                        axs[0].plot([unit_pos[unit_n, 0], unit_pos[unit_n2, 0]],
                            [unit_pos[unit_n, 1], unit_pos[unit_n2, 1]], '-', color = cols[int(w < 0)],
                            zorder = -10, linewidth=.5 + 5 * np.abs(w))

            # plot lines from layer N to the next (N+1)
            for unit_next in nextlay_units:
                w = weights[unit_next, unit_n]
                if w:
                    lines_handles['line_%i-%i' % (unit_next, unit_n)] =\
                        axs[0].plot([unit_pos[unit_next, 0], unit_pos[unit_n, 0]],
                            [unit_pos[unit_next, 1], unit_pos[unit_n, 1]],
                            '-', color = cols[int(w<0)], zorder = -10, linewidth = .5 + 5 * np.abs(w))

    for unit_n in range(n_units):
        axs[2].plot(1, activations[unit_n], 'ko-', markersize = 5)
        for unit_m in range(unit_n, n_units):
            w = weights[unit_m, unit_n]
            axs[1].plot(0, w, 'ko-', markersize = 5, alpha = int(w!=0))
    
    if energy != None:
        axs[2].plot(1, energy, 'ro-', markersize = 7)

    # plot units
    # circles = [pl.Circle(unit_pos[pos_n, :], 0.5/n_units, edgecolor = 'black',
    #                 facecolor=np.repeat(activations[pos_n]/2.25, 3)+.5) for pos_n in range(n_units)]
    circles = [pl.Circle(unit_pos[pos_n, :], 0.4/n_units, edgecolor = 'black',
                    facecolor=[.8, .8, .8]) for pos_n in range(n_units)]
    [axs[0].add_artist(circles[circ_n]) for circ_n in range(len(circles))]

    # make plot square and take off ticks
    axs[0].axis('square'); axs[0].axis([0, 1, 0, 1])
    axs[0].set_yticklabels([]); axs[0].set_xticklabels([])
    axs[0].get_yaxis().set_visible(False); axs[0].get_xaxis().set_visible(False)

    # create the texts
    texts_handles = {}
    for unit_n in range(n_units):
        act_n = activations[unit_n]
        texts_handles['tex_act%i' % (unit_n+1)] = axs[0].text(unit_pos[unit_n, 0], unit_pos[unit_n, 1], '%.2f'%act_n,
        ha='center', va='center', fontsize=10, fontdict={'weight': 'bold', 'family': 'Calibri', 'color' : cols[int(act_n<0)]})#, transform=axs[0].transaxes)

    # texts_handles['tex_delt'] = axs[0].text(.85, 0.95, 'Delta w = 0', ha='center', va='center')#, transform=axs[0].transaxes)
    texts_handles['tex_cycl'] = axs[0].text(.1, 0.95, 'cycle = 1', ha='center', va='center')#, transform=axs[0].transaxes)
    texts_handles['tex_learn_trial_n'] = axs[0].text(.7, 0.95, 'Learning trial n = 0', ha='center', va='center')
    axs[1].set_title('Weights history')
    axs[2].set_title('Activation history')

    fig.canvas.draw()

    return fig, axs, texts_handles, lines_handles, unit_pos



def update_network(fig, axs, texts_handles, lines_handles, activations,
                   unit_pos, weights, layers, change, cycle, energy = None,
                   learn_trial_n = 0):
    '''
    update_network(fig, axs, texts_handles, lines_handles, activations,
                   unit_pos, weights, layers, change, cycle, energy = None)

    Modify a figure plotted with plot_network.
    
    This function returns the figure and axes, the texts and lines handles for
    the network and its unit positions to be used in the update_network function.
    
    Parameters
    ----------
    fig : handle of the figure
    axs : handle of the axes
    texts_handles : handle of the texts (activations and cycle number)
    lines_handles : handle of the lines (weights)
    unit_pos : the position of the units to re-draw weight lines
    activations : array of N activations (floats or integers) for each of the
                  units in the network
    weights : array of N*N numbers (floats or integers) representing the
              connection weights for all units only half of the matrix is used,
              it's always weights[a, b] with a <= b
    layers : an array of integers representing which unit  belongs to which
             layer in the network, e.g. [1, 1, 2] for a 3 unit network with 2
             units on the first layer and one on the second layer.
    energy : (optional) the energy of the network which will be plotted on the
             bottom left subplot in red.
    cycle : (optional) cycle of the updating we are in.
    change : (optional) unused for now.                         
    
    Returns
    -------
        [Nothing]
    '''
    
    n_units = activations.shape[0]
    
    # update the connection lines depending on the new weight matrix
    for lay_n in np.unique(layers):
        lay_units = np.where(layers == lay_n)[0]
        nextlay_units = np.where(layers == lay_n + 1)[0]
        for unit_n in lay_units:
            # plot lines between units of layer N
            for unit_n2 in lay_units:
                w = weights[unit_n, unit_n2]
                if 'line_%i-%i' % (unit_n, unit_n2) in lines_handles.keys():
                    lines_handles['line_%i-%i' % (unit_n, unit_n2)][0].set_alpha(w!=0)
                    if w:
                        lines_handles['line_%i-%i' % (unit_n, unit_n2)][0].set_color(cols[int(w < 0)])
                        lines_handles['line_%i-%i' % (unit_n, unit_n2)][0].set_linewidth(.5 + 5 * np.abs(w))
                else:
                    lines_handles['line_%i-%i' % (unit_n, unit_n2)] =\
                        axs[0].plot([unit_pos[unit_n, 0], unit_pos[unit_n2, 0]],
                            [unit_pos[unit_n, 1], unit_pos[unit_n2, 1]],
                            '-', color = cols[int(w<0)], zorder=-10,
                            linewidth=.5 + 5*np.abs(w))

            # plot lines from layer N to the next (N+1)
            for unit_next in nextlay_units:
                w = weights[unit_next, unit_n]
                if 'line_%i-%i' % (unit_next, unit_n) in lines_handles.keys():
                    lines_handles['line_%i-%i' % (unit_next, unit_n)][0].set_alpha(int(w!=0))
                    if w:
                        lines_handles['line_%i-%i' % (unit_next, unit_n)][0].set_color(cols[int(w < 0)])
                        lines_handles['line_%i-%i' % (unit_next, unit_n)][0].set_linewidth(.5 + 5 * np.abs(w))
                else:
                    lines_handles['line_%i-%i' % (unit_next, unit_n)] =\
                        axs[0].plot([unit_pos[unit_n, 0], unit_pos[unit_next, 0]],
                            [unit_pos[unit_n, 1], unit_pos[unit_next, 1]],
                            '-', color = cols[int(w<0)], zorder=-10,
                            linewidth=.5 + 5*np.abs(w))


    if learn_trial_n != -1:
        texts_handles['tex_learn_trial_n'].set_text('trial n = %i' % learn_trial_n)
    else:
        texts_handles['tex_learn_trial_n'].set_text('')
    texts_handles['tex_cycl'].set_text('optim. cycle = %i' % cycle)

    # update activation texts
    for unit_n in range(n_units):
        texts_handles['tex_act%i' % (unit_n+1)].set_text('%.2f' % activations[unit_n])
        # axs[0].artists[unit_n].set_facecolor(np.repeat(activations[unit_n]/2.25, 3)+.5)

    axs[0].relim()
    axs[0].autoscale_view()

    # update the temporal evolution plot (right)

    lineind = 0
    for unit_n in range(n_units):
        data = axs[2].lines[unit_n].get_data()
        new_x = np.concatenate([data[0], [cycle]])
        new_y = np.concatenate([data[1], [activations[unit_n]]])
        axs[2].lines[unit_n].set_data((new_x, new_y))
        
        for unit_m in range(unit_n, n_units):
            w = weights[unit_m, unit_n]
#            print(w)

            data = axs[1].lines[lineind].get_data()
            new_x = np.concatenate([data[0], [learn_trial_n]])
            new_y = np.concatenate([data[1], [w]])
            axs[1].lines[lineind].set_data((new_x, new_y))
            lineind += 1
    
    if energy != None:
        data = axs[2].lines[-1].get_data()
        new_x = np.concatenate([data[0], [cycle]])
        new_y = np.concatenate([data[1], [energy]])
        axs[2].lines[-1].set_data((new_x, new_y))

    axs[1].relim()
    axs[1].autoscale_view()
    axs[2].relim()
    axs[2].autoscale_view()

    fig.canvas.draw()
    pl.show()
    fig.canvas.flush_events()


def test_plot_network(big = False, timesleep = .5, n_cycles = 10,
                      minimize_links = False):
    #### TEST ####
    # setting up a random network, just to show how it would look if all plotted
    # variables changed at each step
    if big:
        layers = np.array([1, 1, 1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4])
    else:
        layers = np.array([1, 1, 1, 2, 2])
    activations = np.random.random(layers.shape[0])
    n_units = activations.shape[0]
    
    rand_nums = np.random.rand(n_units, n_units)
    weights = np.corrcoef(rand_nums)
    if minimize_links:
        weights[np.random.rand(n_units, n_units)>.4]=0
    
    energy = np.random.random()
    fig, axs, texts_handles, lines_handles, unit_pos =\
        plot_network(figsize = [13, 7], activations = activations,
                      weights = weights, layers = layers, energy = energy)
    
    
    for i in range(1, n_cycles):
        activations = np.random.random(layers.shape[0])
        rand_nums = np.random.rand(n_units, n_units)
        weights = np.corrcoef(rand_nums)
        weights = no_recurrence(weights)
        if minimize_links:
            weights[np.random.rand(n_units, n_units)>.4]=0
        energy = np.random.random()
        change = np.random.random()
    
        update_network(fig = fig, axs = axs, texts_handles = texts_handles,
            lines_handles = lines_handles, activations = activations,
            unit_pos = unit_pos, weights = weights, layers = layers,
            change = change, cycle = i, energy = energy)
        time.sleep(timesleep)






