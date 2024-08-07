'''

This script demonstrates how to build a variational autoencoder with Keras.
Based on
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114


VAE applied to cifar image data (with one hidden layer)
'''
#%% import and initialize
import numpy as np
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
#from scipy.stats import norm
from process_faces import plot_faces

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import metrics , optimizers


#%% main program
if __name__ == "__main__":
	batch_size = 2
	latent_dim = 2
	intermediate_dim = 100 # 256
	epochs = 5000
	epsilon_std = 0.1
	n_faces = 2
	use_vae = True
	n_input = 3
	
	# load and preprocess the stimuli
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	x_train = x_train[:n_faces,:,:,:n_input]
	stim_row, stim_col = x_train.shape[1], x_train.shape[2]
	original_dim = np.prod(x_train.shape[1:])
	print("old shape= ",x_train.shape)
	x_train = x_train.astype('float32') / 255.
	x_train = x_train.reshape((len(x_train), original_dim))
	
	# define the model
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
	decoder_h      = Dense(intermediate_dim, activation='relu')
	decoder_mean   = Dense(original_dim, activation='sigmoid')

	h_decoded      = decoder_h(z)
	x_decoded_mean = decoder_mean(h_decoded)
	
	xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
	kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis = -1 )
	vae_loss = xent_loss + kl_loss # this is now L_B, eq (7) in Kingma & Welling
	
	vae = Model(x, x_decoded_mean)
	if use_vae:
		vae.add_loss(vae_loss)
#		vae.compile(optimizer='rmsprop') # vae loss
		vae.compile(optimizer = 
				    optimizers.legacy.Adam(learning_rate=1e-4))
	else:
	    vae.compile(optimizer='rmsprop', loss="mse") # standard MSE loss for auto-encoder
	
	
	# actual model fitting
	print("new shape = ", x_train.shape)
	history = vae.fit(x_train, x_train,
	        shuffle=True,
	        epochs=epochs,
	        batch_size=batch_size
	        ) # validation_data=(x_test, x_test)
	
	
	# build a model to project inputs on the latent space
	encoder = Model(x, z_mean)
		
	# build a digit generator that can sample from the learned distribution
	decoder_input   = Input(shape=(latent_dim,))
	_h_decoded      = decoder_h(decoder_input)
	_x_decoded_mean = decoder_mean(_h_decoded)
	generator       = Model(decoder_input, _x_decoded_mean)
	
	# print and plot results
	fig = plt.figure()
	plt.plot(history.history["loss"], color = "black")
	plot_faces(generator, stim_row = stim_row, stim_col = stim_col, n_input = n_input)
