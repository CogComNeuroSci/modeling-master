#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 10:05:24 2020

@author: code is adapted from http://lyy1994.github.io/
"""
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import os


def weight(shape, name='weights'):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias(shape, name='biases'):
	return tf.Variable(tf.constant(0.1, shape=shape), name=name)

class RBM:
    i = 0 # flipping index for computing pseudo likelihood
    def __init__(self, n_visible=784, n_hidden=500, k=30, momentum=False):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
		
		# learning rate and momentum
        self.lr = tf.placeholder(tf.float32)
        if momentum:
            self.momentum = tf.placeholder(tf.float32)
        else:
            self.momentum = 0.0
		
		# weights and biases
        self.w  = weight([n_visible, n_hidden], 'w')
        self.hb = bias([n_hidden], 'hb')
        self.vb = bias([n_visible], 'vb')
		
		# velocities of momentum method
        self.w_v  = tf.Variable(tf.zeros([n_visible, n_hidden]), dtype=tf.float32)
        self.hb_v = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32)
        self.vb_v = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)

    def propup(self, visible):
        pre_sigmoid_activation = tf.matmul(visible, self.w) + self.hb
        return tf.nn.sigmoid(pre_sigmoid_activation)

    def propdown(self, hidden):
        pre_sigmoid_activation = tf.matmul(hidden, tf.transpose(self.w)) + self.vb
        return tf.nn.sigmoid(pre_sigmoid_activation)
    
    def sample_h_given_v(self, v_sample):
        h_props = self.propup(v_sample)
        h_sample = tf.nn.relu(tf.sign(h_props - tf.random_uniform(tf.shape(h_props))))
        return h_sample
	
    def sample_v_given_h(self, h_sample):
        v_props = self.propdown(h_sample)
        v_sample = tf.nn.relu(tf.sign(v_props - tf.random_uniform(tf.shape(v_props))))
        return v_sample
    
    def CD_k(self, visibles):       
		# k steps gibbs sampling
        v_samples = visibles
        h_samples = self.sample_h_given_v(v_samples)
        for i in range(self.k):
            v_samples = self.sample_v_given_h(h_samples)
            h_samples = self.sample_h_given_v(v_samples)
		
        h0_props = self.propup(visibles)
        w_positive_grad = tf.matmul(tf.transpose(visibles), h0_props)
        w_negative_grad = tf.matmul(tf.transpose(v_samples), h_samples)
        w_grad = (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(visibles)[0])
        hb_grad = tf.reduce_mean(h0_props - h_samples, 0)
        vb_grad = tf.reduce_mean(visibles - v_samples, 0)
        return w_grad, hb_grad, vb_grad
    
    def learn(self, visibles):
        w_grad, hb_grad, vb_grad = self.CD_k(visibles)
		# compute new velocities
        new_w_v = self.momentum * self.w_v + self.lr * w_grad
        new_hb_v = self.momentum * self.hb_v + self.lr * hb_grad
        new_vb_v = self.momentum * self.vb_v + self.lr * vb_grad
		# update parameters
        update_w = tf.assign(self.w, self.w + new_w_v)
        update_hb = tf.assign(self.hb, self.hb + new_hb_v)
        update_vb = tf.assign(self.vb, self.vb + new_vb_v)
		# update velocities
        update_w_v = tf.assign(self.w_v, new_w_v)
        update_hb_v = tf.assign(self.hb_v, new_hb_v)
        update_vb_v = tf.assign(self.vb_v, new_vb_v)		
        return [update_w, update_hb, update_vb, update_w_v, update_hb_v, update_vb_v]

    def sampler(self, visibles, steps=5000):
        v_samples = visibles
        for step in range(steps):
            v_samples = self.sample_v_given_h(self.sample_h_given_v(v_samples))
        return v_samples
    
    def free_energy(self, visibles):
        first_term = tf.matmul(visibles, tf.reshape(self.vb, [tf.shape(self.vb)[0], 1]))
        second_term = tf.reduce_sum(tf.log(1 + tf.exp(self.hb + tf.matmul(visibles, self.w))), axis=1)
        return - first_term - second_term
    
    def pseudo_likelihood(self, visibles):
        x = tf.round(visibles)
        x_fe = self.free_energy(x)
        split0, split1, split2 = tf.split(x, [self.i, 1, tf.shape(x)[1] - self.i - 1], 1)
        xi = tf.concat([split0, 1 - split1, split2], 1)
        self.i = (self.i + 1) % self.n_visible
        xi_fe = self.free_energy(xi)
        return tf.reduce_mean(self.n_visible * tf.log(tf.nn.sigmoid(xi_fe - x_fe)), axis=0)

import imageio

def save_images(images, size, path):
	img = (images + 1.0) / 2.0
	h, w = img.shape[1], img.shape[2]
	
	merge_img = np.zeros((h * size[0], w * size[1]))
	
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		merge_img[j*h:j*h+h, i*w:i*w+w] = image
		print(image)
	
	return imageio.imsave(path, merge_img)

def train(train_data, epochs):
	# directories to save samples and logs
	logs_dir = './logs'
	samples_dir = './samples'
	
	# markov chain start state
	noise_x, _ = train_data.sample_batch()
	
	# computation graph definition
	x = tf.placeholder(tf.float32, shape=[None, 784])
	rbm = RBM()
	step = rbm.learn(x)
	sampler = rbm.sampler(x)
	pl = rbm.pseudo_likelihood(x)
	
	saver = tf.train.Saver()
    
	# main loop
	with tf.Session() as sess:
		mean_cost = []
		epoch = 1
		init = tf.global_variables_initializer()
		sess.run(init)
		for i in range(epochs * train_data.batch_num):
			# draw samples
			if i % 500 == 0:
				samples = sess.run(sampler, feed_dict = {x: noise_x})
				samples = samples.reshape([train_data.batch_size, 28, 28])
				save_images(samples, [8, 8], os.path.join(samples_dir, 'iteration_%d.png' % i))
				print('Saved samples.')
			batch_x, _ = train_data.next_batch()
			sess.run(step, feed_dict = {x: batch_x, rbm.lr: 0.1})
			cost = sess.run(pl, feed_dict = {x: batch_x})
			mean_cost.append(cost)
			# save model
			if i != 0 and train_data.batch_index == 0:
				checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step = epoch + 1)
				print('Saved Model.')
			# print pseudo likelihood
			if i != 0 and train_data.batch_index == 0:
				print('Epoch %d Cost %g' % (epoch, np.mean(mean_cost)))
				mean_cost = []
				epoch += 1
		
		# draw samples when training finished
		print('Test')
		samples = sess.run(sampler, feed_dict = {x: noise_x})
		samples = samples.reshape([train_data.batch_size, 28, 28])
		save_images(samples, [8, 8], os.path.join(samples_dir, 'test.png'))
		print('Saved samples.') 
    
class DataSet:
	batch_index = 0

	def __init__(self, X, Y, batch_size = None, one_hot = False, seed = 0):
		shape = X.shape
		X = X.reshape([shape[0], shape[1] * shape[2]])
		self.X = X.astype(np.float)/255
		self.size = self.X.shape[0]
		if batch_size == None:
			self.batch_size = self.size
		else:
			self.batch_size = batch_size
		# abandom last few samples
		self.batch_num = int(self.size / self.batch_size)
		# shuffle samples
		np.random.seed(seed)
		np.random.shuffle(self.X)
		np.random.seed(seed)
		np.random.shuffle(Y)
		self.one_hot = one_hot
		if one_hot:
			y_vec = np.zeros((len(Y), 10), dtype=np.float)
			for i, label in enumerate(Y):
				y_vec[i, Y[i]] = 1.0
			self.Y = y_vec
		else:
			self.Y = Y	
	def next_batch(self):
		start = self.batch_index * self.batch_size
		end = (self.batch_index + 1) * self.batch_size
		self.batch_index = (self.batch_index + 1) % self.batch_num
		if self.one_hot:
			return self.X[start:end, :], self.Y[start:end, :]
		else:
			return self.X[start:end, :], self.Y[start:end]
	def sample_batch(self):
		index = np.random.randint(self.batch_num)
		start = index * self.batch_size
		end = (index + 1) * self.batch_size
		if self.one_hot:
			return self.X[start:end, :], self.Y[start:end, :]
		else:
			return self.X[start:end, :], self.Y[start:end]	

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_data = DataSet(x_train, y_train, batch_size = 10)

train(train_data, 5)


