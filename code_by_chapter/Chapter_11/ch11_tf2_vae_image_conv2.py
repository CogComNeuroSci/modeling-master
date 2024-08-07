'''

This script demonstrates how to build a variational autoencoder with Keras.
Based on
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

This code is originally based on code by prl900:
https://gist.github.com/prl900/a423a1b021ed8d4ae78b0311e371e559

VAE on image data, with 2 convolutional layers
'''
#%% import and initialize
import numpy as np
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from process_faces import plot_faces

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape,\
	 Conv2DTranspose, AveragePooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import metrics, optimizers


if __name__ == "__main__":
	batch_size = 2
	latent_dim = 16
	hidden_dim = 32 # 256
	epochs = 100000
	epsilon_std = 0.1
	n_stimuli = 2
	n_input_filter, n_filter1, n_filter2 = 3, 4, 8
	
	# load and preprocess the stimuli
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	x_train = x_train[:n_stimuli,:,:,:n_input_filter]
	print(x_train.shape)
	stim_row = x_train.shape[1]
	stim_col = x_train.shape[2]
	original_dim = np.prod(x_train.shape[1:])
	x_train = x_train.astype('float32') / 255.
	
	# the loss function
	def loss():
		input_reshaper = Reshape((stim_row*stim_col*n_input_filter,))
		x_reshaped = input_reshaper(x) 
		x_decoded_mean_reshaped = input_reshaper(x_decoded_mean)
		xent_loss = original_dim * metrics.binary_crossentropy(x_reshaped, x_decoded_mean_reshaped)
		kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis = -1 )
		the_loss =  xent_loss + kl_loss # this is now L_B, eq (7) in Kingma & Welling
		return the_loss

	# define the encoder model
	x                 = Input(batch_shape=(batch_size, stim_row, stim_col, n_input_filter))

	c1                = Conv2D(filters = n_filter1, padding = "same", kernel_size = (3, 3), input_shape = (x_train.shape[1], x_train.shape[2], n_input_filter))(x)
	conv_dim1         = c1.get_shape()[1:3] # sizes of the filters
	c1_sub            = AveragePooling2D(pool_size = (1, 1))(c1)
	sub_conv_dim1     = c1_sub.get_shape()[1:3]
	c1_sub_bn         = BatchNormalization()(c1_sub)

	c2                = Conv2D(filters = n_filter2, padding = "same", kernel_size = (3, 3))(c1_sub_bn)
	conv_dim2         = c2.get_shape()[1:3] # sizes of the filters
	c2_sub            = AveragePooling2D(pool_size = (1, 1))(c2)
	sub_conv_dim2     = c2_sub.get_shape()[1:3]
	c2_sub_bn         = BatchNormalization()(c2_sub)

	cf     = Flatten()(c2_sub_bn)
	convolution_dim = cf.shape[1] # dimension of the flattened, subsampled conv layer
	h      = Dense(hidden_dim, activation='relu')(cf)
	
	z_mean = Dense(latent_dim, activation='relu')(h)
	z_log_var = Dense(latent_dim, activation='relu')(h)
	epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0.,
	                              stddev=epsilon_std)
	z = z_mean + tf.math.exp(z_log_var / 2) * epsilon
	
	# define the decoder model; we instantiate these layers separately so as to reuse them later for the generator
	decoder_h      = Dense(hidden_dim, activation='relu')
	decoder_cf     = Dense(convolution_dim, activation='relu')
	
	reshaper2      = Reshape((sub_conv_dim2[0], sub_conv_dim2[1], n_filter2))
	upsampler2     = UpSampling2D(size = (1, 1))
	bn2            = BatchNormalization()
	
	decoder_c1     = Conv2DTranspose(filters = n_filter1, padding = "same", kernel_size = (3, 3))
	upsampler1     = UpSampling2D(size = (1, 1)) 
	bn1            = BatchNormalization()

	decoder_mean   = Conv2DTranspose(filters = n_input_filter, padding = "same", kernel_size = (3, 3))

	h_decoded      = decoder_h(z)
	cf_decoded     = decoder_cf(h_decoded)
	c2_sub_reshaped= reshaper2(cf_decoded)
	c2_reshaped    = upsampler2(c2_sub_reshaped) # upsampling
	c2_bn          = bn2(c2_reshaped)

	c1_sub_reshaped= decoder_c1(c2_bn)
	c1             = upsampler1(c1_sub_reshaped)
	c1_bn          = bn1(c1)
	x_decoded_mean = decoder_mean(c1_bn)

	vae = Model(x, x_decoded_mean)
	vae.add_loss(loss())
#	vae.compile(optimizer='rmsprop') # vae loss
	vae.compile(optimizer = 
			    optimizers.legacy.Adam(learning_rate=1e-3))
		
	#%% actual model fitting
	print("new shape = ", x_train.shape)
	history = vae.fit(x_train, x_train,
	        shuffle=True,
	        epochs=epochs,
	        batch_size=batch_size
	        ) # validation_data=(x_test, x_test)
	
	# build a model to project inputs on the latent space
	encoder = Model(x, z_mean)
	
	# build a face generator that can sample from the learned distribution
	decoder_input    = Input(shape=(latent_dim,))
	_h_decoded       = decoder_h(decoder_input)
	_cf_decoded      = decoder_cf(_h_decoded)
	_c2_sub_reshaped = reshaper2(_cf_decoded)
	_c2_reshaped     = upsampler2(_c2_sub_reshaped)
	_c2_bn           = bn2(_c2_reshaped)

	_c1_decoded_mean = decoder_c1(_c2_bn)
	_c1_upsampled    = upsampler1(_c1_decoded_mean)
	_c1_bn           = bn1(_c1_upsampled)
	_x_decoded_mean  = decoder_mean(_c1_bn)
	
	generator        = Model(decoder_input, _x_decoded_mean)

	#%% print and plot results
	fig = plt.figure()
	plt.plot(history.history["loss"], color = "black")
	plot_faces(generator, stim_row = stim_row, stim_col = stim_col, n_input= n_input_filter)
