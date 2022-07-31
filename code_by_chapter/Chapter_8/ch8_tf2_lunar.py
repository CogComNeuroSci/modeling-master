#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:38:02 2022

@author: tom verguts
solves the lunar lander problem using policy gradient (-like) algorithm (reinforce)
it works but it's inefficient 
"""
#%% import, initialization, definitions
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ch8_tf2_pole_1 import perform
import os
import tensorflow.keras.backend as K


class PG_Agent(object):
    def __init__(self, n_states, n_actions, lr, gamma, max_n_step):
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions = np.arange(n_actions)
        self.lr = lr
        self.gamma = gamma
        self.max_n_step = max_n_step
        self.n_step = max_n_step
        self.x_buffer = np.zeros((self.max_n_step, n_states))
        self.y_buffer = np.zeros((self.max_n_step, 2))
        self.network = self.build_network()

    def build_network(self):
        def PG_loss(y_true, y_pred):
            action_true = K.cast(y_true[:, 0], "int32")
            advantage =   y_true[:, 1]
            pred = K.clip(y_pred, 1e-8, 1-1e-8)
            sum_total = 0 # clunky workaround bcs direct solution didn't work...
            for loop in range(self.n_step):
                sum_total +=  advantage[loop]*K.log(pred[loop, action_true[loop]])
            return -sum_total
#           return -K.log(y_pred[: , action_true] + 1e-5) * advantage

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
        self.y_buffer = np.zeros((self.max_n_step, 2))		
		
    def update_buffer(self, n_step, states, actions, rewards):
        self.x_buffer = states
        self.y_buffer[:, 0] = np.squeeze(actions).astype(int)
        self.n_step = n_step
        for indx in range(n_step):
            weighted_reward = 0
            gamma_w = 1 # re_initialize
            for loop in np.arange(indx, n_step):
                weighted_reward += gamma_w*rewards[loop]
                gamma_w *= self.gamma # discount
            self.y_buffer[indx, 1] = weighted_reward
        avg = np.mean(self.y_buffer[:n_step, 1])
        std = np.std(self.y_buffer[:n_step, 1])
        self.y_buffer[:, 1] = (self.y_buffer[:, 1] - avg)/(std + int(std == 0))  	    
            
    def learn(self, verbose: bool = True):
       	self.network.train_on_batch(self.x_buffer, self.y_buffer)	
        if verbose:
            print("what do you want to see?")
 
    def sample(self, state):
        state_array = np.array(state)[np.newaxis, :]
        prob = np.squeeze(self.network.predict(state_array))
        action = np.random.choice(self.actions, p = prob) 
        return action

    
def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4, success_crit: int = 10):
    lc = np.zeros(n_loop)
    stop_crit = False
    loop, success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
        states  = np.zeros((max_n_step, env.observation_space.shape[0]))
        actions = np.zeros(max_n_step) # to construct y_buffer
        rewards = np.zeros(max_n_step) # to construct y_buffer
        while not done and n_step < max_n_step:
            action = rl_agent.sample(state)
            next_state, reward, done, info = env.step(action)
            states[n_step, :]  = state
            actions[n_step] = action
            rewards[n_step] = reward
            n_step += 1
            state = next_state
        # action done, now learn
        rl_agent.empty_buffer()
        rl_agent.update_buffer(n_step, states, actions, rewards)
        rl_agent.learn(verbose = False)
        lc[loop] = np.max(rewards)
        loop += 1
        success += int(np.max(rewards) >= 200)
        rw = np.max(rewards[:n_step])
        print("n steps = " + str(n_step) + " , max rew = {:.1f}".format(rw) + "\n" )
        stop_crit = (loop == n_loop) or (success > success_crit)
    return lc, success > success_crit

#%% main code
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    load_model, save_model, train_model, performance = False, False, True, True
    rl_agent = PG_Agent(n_states = env.observation_space.shape[0], n_actions = env.action_space.n, \
                           lr = 0.0005, gamma = 0.99, max_n_step = 600)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/models"+"/model_lunar", compile = False)
    if train_model:
        lc, solved = learn_w(env, n_loop = 1500, input_dim = env.observation_space.shape[0], max_n_step = rl_agent.max_n_step)
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/models"+"/model_lunar.h5")
    if train_model:
        plt.plot(lc)
    if train_model and solved:
        print("Problem solved.")
    if performance:
        perform(env, rl_agent, verbose = False)
    env.close()
