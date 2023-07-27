#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:38:02 2022

@author: tom verguts
solves the lunar lander problem using actor-critic
and policy gradient for the actor 
"""
#%% import, initialization, definitions
import gym
import numpy as np
import tensorflow as tf
from ch8_tf2_pole_1 import perform
from ch8_tf2_lunar import PG_Agent
from ch8_tf2_taxi_2 import plot_data
import os
import tensorflow.keras.backend as K


class PG_Agent_AC(PG_Agent):
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
        self.network = self.build_network(n_out = self.n_actions, loss = self.PG_loss)
        self.critic = self.build_network(n_out = 1, loss = tf.keras.losses.MeanSquaredError())

    def PG_loss(self, y_true, y_pred):
        """TD-like loss function for the actor (network)"""
        action_true = K.cast(y_true[:, 0], "int32")
        pred = K.clip(y_pred, 1e-8, 1-1e-8)
        advantage =   y_true[:, 1]
        sum_total = 0 # clunky workaround bcs direct solution didn't work...
        for loop in range(self.n_step):
            sum_total +=  advantage[loop]*K.log(pred[loop, action_true[loop]])
        return -sum_total

    def build_network(self, n_out, nhid1: int = 64, nhid2: int = 64, loss = tf.keras.losses.MeanSquaredError()):
        model = tf.keras.Sequential([ 
                tf.keras.layers.Dense(nhid1, input_shape = (self.n_states,), activation = "relu"),
                tf.keras.layers.Dense(nhid2, activation = "relu"),
                tf.keras.layers.Dense(n_out, activation = "softmax")
		    	] )
        model.build()
        model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = self.lr), loss = loss)
        return model
 
    def update_buffer(self, n_step, states, actions, rewards):
        self.x_buffer = states
        self.y_buffer[:, 0] = np.squeeze(actions).astype(int)
        self.y_buffer[:, 1] = rewards
        self.n_step = n_step
        indx = np.arange(self.n_step)
        indx_next = indx + (indx<self.n_step-1)
        x_next = self.x_buffer[indx_next]
        v = self.critic.predict(self.x_buffer[:n_step])
        v_next = np.squeeze(self.critic.predict(x_next))
        self.y_buffer[:self.n_step, 1] = \
               np.squeeze(rewards[:n_step]) + self.gamma*v_next*np.squeeze(indx < self.n_step - 1) - np.squeeze(v)
#        avg = np.mean(self.y_buffer[:n_step, 1])
#        std = np.std(self.y_buffer[:n_step, 1])
#        self.y_buffer[:, 1] = (self.y_buffer[:, 1] - avg)/(std + int(std == 0))  	    
            
    def learn(self, n_step, verbose: bool = True):
       	self.network.train_on_batch(self.x_buffer, self.y_buffer)
        rewards = self.y_buffer[:, 1]
        indx = np.arange(self.n_step)
        indx_next = indx + (indx<self.n_step-1)
        x_next = self.x_buffer[indx_next]
        v_next = np.squeeze(self.critic.predict(x_next))
        value_estimate = \
               np.squeeze(rewards[:n_step]) + self.gamma*v_next*np.squeeze(indx < self.n_step - 1)   
        self.critic.train_on_batch(self.x_buffer[:n_step], value_estimate)
        if verbose:
            print("what do you want to see?")

def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4, success_crit: int = 10):
    lc = np.zeros(n_loop)
    reward_vec = np.zeros(n_loop)
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
        rl_agent.learn(n_step, verbose = False)
        lc[loop] = n_step
        reward_vec[loop] = np.max(rewards)
        loop += 1
        success += int(np.max(rewards) >= 200)
        rw = np.max(rewards[:n_step])
        print("n steps = " + str(n_step) + " , max rew = {:.1f}".format(rw) + "\n" )
        stop_crit = (loop == n_loop) or (success > success_crit)
    return lc, reward_vec, success > success_crit

#%% main code
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    load_model, save_model, train_model, performance = False, False, True, True
    rl_agent = PG_Agent_AC(n_states = env.observation_space.shape[0], n_actions = env.action_space.n, \
                           lr = 0.0005, gamma = 0.99, max_n_step = 1000)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/models"+"/model_lunar", compile = False)
    if train_model:
        lc, reward_vec, solved = learn_w(env, n_loop = 30, input_dim = env.observation_space.shape[0], max_n_step = rl_agent.max_n_step)
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/models"+"/model_lunar.h5")
    if train_model:
        plot_data(10, reward_vec, lc)
    if train_model and solved:
        print("Problem solved.")
    if performance:
        perform(env, rl_agent, verbose = False)
    env.close()
