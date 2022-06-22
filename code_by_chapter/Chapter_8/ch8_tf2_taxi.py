#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:38:02 2022

@author: tom verguts
solves the taxi problem using policy gradient (-like) algorithm
"""
#%% import, initialization, definitions
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ch8_tf2_pole_1 import perform



class PG_Agent(object):
    def __init__(self, n_states, n_actions, lr, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions = np.arange(n_actions)
        self.lr = lr
        self.gamma = gamma
        self.network = build_network(self.n_states, self.n_actions, self.lr)

    def build_network(self):
    	def PG_loss(y_true, y_pred):
            action_true = y_true[:, :action_dim]
            advantage = y_true[:, action_dim:]
            return -np.log(y_pred.prob(action_true) + 1e-5) * advantage

        model = tf.keras.Sequential([ 
                tf.keras.layers.Dense(action_dim, activation = "linear", name = "layer")
		    	] )
        model.build()
        loss = {"layer": PG_loss()}
        model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss)
        return model
 
    def empty_buffer():
		self.x_buffer = np.zeros((max_nstep, self.n_states))
		self.y_buffer[:, actions] = np.zeros((max_nstep, self.n_actions + 1))		
		
    def update_buffer(n_step, states, actions, rewards):
		self.x_buffer = states
		for indx, el in actions:
			self.y_buffer[indx, el] = 1
		for indx in n_step:
            weighted_reward = 0
			gamma_w = self.gamma # gamma raised to some power
			for loop in np.arange(indx, n_step):
    		    weighted_reward += gamma_w*rewards[loop]
				gamma_w *= self.gamma
  		    self.y_buffer[indx, -1] = weighted_reward
		
    def learn(self, verbose: bool = True):
       	self.network.fit(self.x_buffer, self.y_buffer, verbose = 0)	
        if verbose:
			print("something")
           
    def sample(self, state):
        y = self.network.predict(np.array(state[np.newaxis,:]))
        prob = np.scipy.softmax(y)
        action = np.choice(self.actions, prob) 
        return action


def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4):
    lc = np.zeros(n_loop)
    buffer_count = 0
    stop_crit = False
    loop, success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
		states  = np.zeros((max_n_step, env.input_space.shape[0]))
        actions = np.zeros((max_n_step, 1)) # to construct y_buffer
		rewards = np.zeros((max_n-step, 1))
        for t in range(max_per_episode):
            action = rl_agent.sample(state)
            next_state, reward, done, info = env.step(action)
            states[n_step, :]  = state
            actions[n_step, :] = action
            rewards[n_step, :] = reward
            n_step += 1
			if done:
			    break
        rl_agent.empty_buffer()
		rl_agent.update_buffer(n_step, states, actions, rewards)
        rl_agent.learn(verbose = False)
        lc[loop] = n_step
        loop += 1
        success += int(done)
        print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (success > 10)
    return lc, success > 10

#%% main code
if __name__ == "__main__":
    env = gym.make('Taxi-v2')
    load_model, save_model, train_model = False, False, True
    rl_agent = PG_Agent(env.observation_space.shape[0], env.action_space.n, \
                           lr = 0.1, gamma = 0.9)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/model_taxi")
    if train_model:
        lc, solved = learn_w(env, n_loop = 100, max_n_step = 200, input_dim = env.observation_space.shape[0])
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/model_taxi")
    if train_model:
        plt.plot(lc)
    if train_model and solved:
        print("Problem solved.")
    perform(env, rl_agent, verbose = False)
    env.close()
