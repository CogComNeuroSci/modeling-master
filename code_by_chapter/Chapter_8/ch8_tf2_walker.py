#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 17:48:43 2022

@author: tom verguts
a policy gradient (PG) approach to (continuous-action) bipedal walker
it's a hard problem...
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tensorflow.keras.backend as K
from os.path import join
from ch8_tf2_mountaincar_cont import learn_w, perform
from ch9_RL_taxi import smoothen

class PG_Agent(object):
    def __init__(self, n_states, n_actions, lr, gamma, max_n_step):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_real_actions = self.n_actions//2
        self.actions = np.arange(n_actions)
        self.lr = lr
        self.gamma = gamma
        self.max_n_step = max_n_step
        self.x_buffer = np.zeros((self.max_n_step, n_states))
        self.y_buffer = np.zeros((self.max_n_step, self.n_actions + 1))
        self.network = self.build_network()

    def build_network(self):
        def PG_loss(y_true, y_pred):
            action_true = y_true[:, 0:self.n_real_actions]
            advantage =   y_true[:, -1]
            pred = K.clip(y_pred, 1e-8, 1-1e-8)
            sum_total = 0 # computing n_real_actions log normal densities for each step.. pred is mean and log std
            for loop in range(self.max_n_step):
                for action_loop in range(self.n_real_actions):
                 sum_total +=  \
                    advantage[loop]*(-pred[loop, (self.n_actions%self.n_real_actions)*action_loop + 1]
                    -0.5*K.square( (action_true[loop]-pred[loop, (self.n_actions%self.n_real_actions)*action_loop + 0])/
                                    K.exp(pred[loop, (self.n_actions%self.n_real_actions)*action_loop + 1]) ))
            return -sum_total

        model = tf.keras.Sequential([ 
                tf.keras.layers.Dense(64, input_shape = (self.n_states,), activation = "relu"),
                tf.keras.layers.Dense(64, activation = "relu"),
                tf.keras.layers.Dense(self.n_actions, activation = "softmax")
		    	] )
        model.build()
        model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = self.lr), loss = PG_loss)
        return model
 
    def empty_buffer(self):
        self.x_buffer = np.zeros((self.max_n_step, self.n_states))
        self.y_buffer = np.zeros((self.max_n_step, self.n_real_actions + 1))
		
    def update_buffer(self, n_step, states, actions, rewards):
        self.x_buffer = states
        self.y_buffer[:, :self.n_real_actions] = actions
        for indx in range(n_step):
            weighted_reward = 0
            gamma_w = 1 # re_initialize
            for loop in np.arange(indx, n_step):
                weighted_reward += gamma_w*rewards[loop]
                gamma_w *= self.gamma # discount
            self.y_buffer[indx, -1] = weighted_reward
        avg = np.mean(self.y_buffer[:n_step, -1])
        std = np.std(self.y_buffer[:n_step, -1])
        self.y_buffer[:, -1] = (self.y_buffer[:, -1] - avg)/(std + int(std == 0))  	    
            
    def learn(self, verbose: bool = True):
       	self.network.train_on_batch(self.x_buffer, self.y_buffer)	
        if verbose:
            print("what do you want to see?")
 
    def sample(self, state):
        pars = rl_agent.network.predict(state[np.newaxis,:])
        action = np.random.normal(loc = pars[0, 0::2], scale = np.exp(pars[:, 1::2])) 
        return np.minimum( np.maximum(-1+1e-5, action), 1-1e-5)



if __name__ == "__main__":
    env = gym.make("BipedalWalker-v2")
    load_model, save_model, train_model = False, False, True
    rl_agent = PG_Agent(n_states = env.observation_space.shape[0], n_actions = 2*4, \
                           lr = 0.0005, gamma = 0.99, max_n_step = 400)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(join(os.getcwd(), "models", "model_walker.h5"))
    if train_model:
        lc, solved = learn_w(env, rl_agent, n_loop = 2000, max_n_step = rl_agent.max_n_step, input_dim = env.observation_space.shape[0],
                                  aggr = np.sum, n_actions = rl_agent.n_real_actions, success_crit_episode = 100)
    if save_model:
        tf.keras.models.save_model(rl_agent.network, join(os.getcwd(), "models", "model_walker.h5"))
    if train_model:
        plt.plot(smoothen(lc, 10))
    if train_model and solved:
        print("Problem solved.")
    perform(env, rl_agent, verbose = False)
    env.close()