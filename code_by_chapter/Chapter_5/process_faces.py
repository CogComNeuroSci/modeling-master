#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:51:31 2024

@author: tom verguts
functions for loading and preprocessing the face data
used in ch5_tf2_face_classif_conv.py
"""

from os import chdir, listdir, getcwd, path
import numpy as np
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_faces(generator, n: int = 15, stim_row: int = 172, stim_col: int = 245, show: bool = True, dims: tuple = (0, 1)):
	# display a 2D manifold of the stimuli (e.g., digits)
	input_dim = generator.inputs[0].shape[1]
	n = 5
	if input_dim > 1:
		n1 = n  # figure with n x n digits
	else:
		n1 = 1
	figure = np.zeros((stim_row * n1, stim_col * n))
	# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
	# to produce values of the latent variables z, since the prior of the latent space is Gaussian
	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	if input_dim > 1:
		grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
	if input_dim > 1:
		for i, yi in enumerate(grid_x):
			for j, xi in enumerate(grid_y):
				z_sample = np.zeros((1, input_dim))
#				data = np.array([[xi, yi]])
				z_sample[0, dims[0]] = xi
				z_sample[0, dims[1]] = yi
#				if input_dim > 2:
#					z_sample = np.column_stack((z_sample, np.zeros((1, input_dim-2))))
				x_decoded = generator.predict(z_sample, verbose = 0)
				face = x_decoded[0].reshape(stim_row, stim_col)
				figure[i * stim_row: (i + 1) * stim_row,
		               j * stim_col: (j + 1) * stim_col] = face
	else:
		for i, yi in enumerate(grid_x):
			z_sample = np.array([[yi]])
			x_decoded = generator.predict(z_sample, verbose = 0)
			face = x_decoded[0].reshape(stim_row, stim_col)
			figure[0: stim_row,
                i * stim_col: (i + 1) * stim_col] = face
	if show:
		plt.figure(figsize=(10, 10))
		plt.imshow(figure, cmap='Greys_r')
		plt.show()
	return figure	

def load_faces(dirs, gran = 10, n_faces = 2):
# gran = granularity, amount of downsampling of the stimuli
	dirs = path.join(getcwd(), dirs)
	chdir(dirs)
	list_faces = listdir(dirs)
	faces = []
	sample = 0
	n_face = 0
	while True:
		file = path.join(dirs, list_faces[sample])
		sample += 1
		if file[-3:] == "jpg":
			n_face += 1
			faces.append(imread(file)[::gran,::gran,0])
		if n_face == n_faces:
			break
	faces = np.array(faces)	
	return faces

def load_faces_labels(dirs, gran = 10, n_faces = 2, test = 0.2, depth = 1):
# gran = granularity, amount of downsampling of the stimuli
# here the label (e.g., male/female is also added to the data)
# test = proportion (0-1 scale) of all data for test data 
	gender_location = -15
	dirs = path.join(getcwd(), dirs)
	chdir(dirs)
	list_faces = listdir(dirs)
	faces = []
	labels= []
	sample = 0
	n_face = 0
	while True:
		file = path.join(dirs, list_faces[sample])
		sample += 1
		if file[-3:] == "jpg":
			n_face += 1
			faces.append(imread(file)[::gran,::gran,:depth])
			labels.append(int(file[gender_location]=="M"))
		if n_face == n_faces:
			break
	faces = np.array(faces)	
	labels= np.array(labels)
	if test > 0:
		nrs = np.arange(n_faces)
		np.random.shuffle(nrs)
		n_train   = int(np.floor(n_faces*(1-test)))
		faces_train = faces[nrs[:n_train]]
		faces_test  = faces[nrs[n_train:]]
		labels_train= labels[nrs[:n_train]]
		labels_test = labels[nrs[n_train:]]
	else:
		faces_train = faces
		faces_test  = faces
		labels_train= labels
		labels_test = labels
	del faces				
	return faces_train, labels_train, faces_test, labels_test

def show_face(face):
	plt.figure(figsize=(10, 10))
	plt.imshow(face, cmap='Greys_r')
	plt.show()
	
if __name__ == "__main__":
	# just a sanity check
	faces = load_faces("CFD-Version-3.0/Images/CFD-INDIA", gran = 30, n_faces = 20)
	print(faces.shape)
	show_face(faces[1])
