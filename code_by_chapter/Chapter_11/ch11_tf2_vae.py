'''

This script demonstrates how to build a variational autoencoder with Keras.
Based on
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

This code is a slightly adapted  version of code by prl900:
https://gist.github.com/prl900/a423a1b021ed8d4ae78b0311e371e559

Things to try:
	- Compare the free energy (VAE) loss with the standard MSE loss discussed in the MCP.
	Do you see differences?
	- Give different weights to the reconstruction (xent) and complexity (kl) loss parts of the VAE.
	This corresponds to a beta-VAE as described in Higgins et al (2017, Arxiv).
	What happens if you change the weights?
'''
#%% import and initialize
import numpy as np
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist

def plot_digits(generator, n: int = 15, digit_row: int = 28, digit_col: int = 28, show: bool = True):
	# display a 2D manifold of the stimulu (e.g., digits)
	input_dim = generator.inputs[0].shape[1]
	n = 15
	if input_dim > 1:
		n1 = n  # figure with 15x15 digits
	else:
		n1 = 1
	figure = np.zeros((digit_row * n1, digit_col * n))
	# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
	# to produce values of the latent variables z, since the prior of the latent space is Gaussian
	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	if input_dim > 1:
		grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
	if input_dim > 1:
		for i, yi in enumerate(grid_x):
			for j, xi in enumerate(grid_y):
				z_sample = np.array([[xi, yi]])
				if input_dim > 2: 
					z_sample = np.column_stack((z_sample, np.zeros((1, input_dim-2))))
				x_decoded = generator.predict(z_sample)
				digit = x_decoded[0].reshape(digit_row, digit_col)
				figure[i * digit_row: (i + 1) * digit_row,
		               j * digit_col: (j + 1) * digit_col] = digit
						
	else:
		for i, yi in enumerate(grid_x):
			z_sample = np.array([[yi]])
			if input_dim > 1: 
				z_sample = np.column_stack((z_sample, np.zeros((1, input_dim-1))))
			x_decoded = generator.predict(z_sample)
			digit = x_decoded[0].reshape(digit_row, digit_col)
			figure[0: digit_row,
                i * digit_col: (i + 1) * digit_col] = digit
	if show:
		plt.figure(figsize=(10, 10))
		plt.imshow(figure, cmap='Greys_r')
		plt.show()
	return figure	


if __name__ == "__main__":
	batch_size = 50
	original_dim = 784
	latent_dim = 1
	intermediate_dim = 256
	epochs = 30
	epsilon_std = 1
	use_vae = True
	
	x = Input(batch_shape=(batch_size, original_dim))
	h = Dense(intermediate_dim, activation='relu')(x)
	z_mean = Dense(latent_dim)(h)
	z_log_var = Dense(latent_dim)(h)
	
	def sampling(args):
	    z_mean, z_log_var = args
	    epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0.,
	                              stddev=epsilon_std)
	    return z_mean + tf.math.exp(z_log_var / 2) * epsilon
	
	z = sampling([z_mean, z_log_var])
	
	# we instantiate these layers separately so as to reuse them later
	decoder_h = Dense(intermediate_dim, activation='relu')
	decoder_mean = Dense(original_dim, activation='sigmoid')
	h_decoded = decoder_h(z)
	x_decoded_mean = decoder_mean(h_decoded)
	
	xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
	kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis = -1 )
	vae_loss = xent_loss + kl_loss # this is now L_B, eq (7) in Kingma & Welling
	
	vae = Model(x, x_decoded_mean)
	if use_vae:
	    vae.add_loss(vae_loss)
	    vae.compile(optimizer='rmsprop') # vae loss
	else:
	    vae.compile(optimizer='rmsprop', loss="mse") # standard MSE loss for auto-encoder
	
	# train the VAE on MNIST digits
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	subset = np.arange(10)
#	subset = np.array([0, 1])
	x_train= x_train[np.isin(y_train, subset)]
	nrem =   x_train.shape[0]%batch_size
	x_train = x_train.astype('float32') / 255.
	x_test  = x_test[np.isin(y_test, subset)]	
	x_test = x_test.astype('float32') / 255.
	if nrem>0:
		x_train = x_train[:-nrem]
		x_test  = x_test[:-nrem]
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	
	#%% actual model fitting
	print(x_train.shape, x_test.shape)
	vae.fit(x_train, x_train,
	        shuffle=True,
	        epochs=epochs,
	        batch_size=batch_size
	        ) # validation_data=(x_test, x_test)
	
	
	# build a model to project inputs on the latent space
	encoder = Model(x, z_mean)
	
#  	# display a 2D plot of the digit classes in the latent space
# 	x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# 	plt.figure(figsize=(6, 6))
# 	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# 	plt.colorbar()
# 	plt.show()
	
	# build a digit generator that can sample from the learned distribution
	decoder_input = Input(shape=(latent_dim,))
	_h_decoded = decoder_h(decoder_input)
	_x_decoded_mean = decoder_mean(_h_decoded)
	generator = Model(decoder_input, _x_decoded_mean)
	
	#%% print and plot results
	plot_digits(generator)
