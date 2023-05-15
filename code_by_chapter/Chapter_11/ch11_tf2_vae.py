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

def plot_digits(generator, n: int = 15, digit_size: int = 28):
	# display a 2D manifold of the digits
	n = 15  # figure with 15x15 digits
	digit_size = 28
	figure = np.zeros((digit_size * n, digit_size * n))
	# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
	# to produce values of the latent variables z, since the prior of the latent space is Gaussian
	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			z_sample = np.array([[xi, yi]])
			x_decoded = generator.predict(z_sample)
			digit = x_decoded[0].reshape(digit_size, digit_size)
			figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
	plt.figure(figsize=(10, 10))
	plt.imshow(figure, cmap='Greys_r')
	plt.show()

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
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

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#%% actual model fitting
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

#%% print and plot results
plot_digits(generator)

