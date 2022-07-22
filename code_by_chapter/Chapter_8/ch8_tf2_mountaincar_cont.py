#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:38:02 2022

@author: tom verguts
first attempt at continuous mountain car problem with policy gradient (PG)
didn't bother to import an earlier class here, too many changes
"""

#%% import, initialization, definitions
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
import tensorflow.keras.backend as K
sys.path.append('/Users/tom/Documents/Modcogproc/modeling-master/code_by_chapter/Chapter_9')
from ch9_RL_taxi import smoothen


class PG_Agent(object):
    def __init__(self, n_states, n_actions, lr, gamma, max_n_step):
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions = np.arange(n_actions)
        self.lr = lr
        self.gamma = gamma
        self.max_n_step = max_n_step
        self.x_buffer = np.zeros((self.max_n_step, n_states))
        self.y_buffer = np.zeros((self.max_n_step, 2))
        self.network = self.build_network()

    def build_network(self):
        def PG_loss(y_true, y_pred):
            action_true = y_true[:, 0]
            advantage =   y_true[:, 1]
            pred = K.clip(y_pred, 1e-8, 1-1e-8)
            sum_total = 0 # computing a log normal density here...
            for loop in range(self.max_n_step):
                sum_total +=  \
                    advantage[loop]*(-pred[loop, 1]
                    -0.5*K.square( (action_true[loop]-pred[loop, 0])/K.exp(pred[loop, 1]) ))
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
        self.y_buffer = np.zeros((self.max_n_step, 2))		
		
    def update_buffer(self, n_step, states, actions, rewards):
        self.x_buffer = states
        self.y_buffer[:, 0] = np.squeeze(actions)
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
        pars = rl_agent.network.predict(state[np.newaxis,:])
        action = np.random.normal(loc = pars[0, 0], scale = np.exp(pars[0, 1])) 
        return np.minimum( np.maximum(-1+1e-5, action), 1-1e-5)

    
def learn_w(env, rl_agent: PG_Agent, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4, 
                 aggr = np.max, n_actions: int = 1, success_crit_episode: int = 200, success_crit: int = 10):
    lc = np.zeros(n_loop)
    stop_crit = False
    loop, success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
        states  = np.zeros((max_n_step, env.observation_space.shape[0]))
        actions = np.zeros((max_n_step, n_actions)) # to construct y_buffer
        rewards = np.zeros(max_n_step) # to construct y_buffer
        for t in range(max_n_step):
            action = rl_agent.sample(state)
            if len(action.shape) > 1:
                action = action[0]
            else:
                action = [action]
            next_state, reward, done, info = env.step(action)
            states[n_step, :]  = state
            actions[n_step] = action
            rewards[n_step] = reward
            n_step += 1
            state = next_state
            if done:
                break
        rl_agent.empty_buffer()
        rl_agent.update_buffer(n_step, states, actions, rewards)
        rl_agent.learn(verbose = False)
        lc[loop] = aggr(rewards)
        rw = lc[loop] 
        loop += 1
        success += int(np.max(rewards) >= success_crit_episode)
        print("n steps = " + str(n_step) + " , aggr rew = {:.1f}".format(rw) + "\n" )
        stop_crit = (loop == n_loop) or (success > success_crit)
    return lc, success > success_crit

def perform(env, rl_agent, verbose: bool = False):
    state = env.reset()
    n_step, done = 0, False
    while not done:
        env.render()
        action = rl_agent.sample(state)
        if len(action.shape) > 1:
            action = action[0]
        else:
            action = [action]
        next_state, reward, done, info = env.step(action)
        n_step += 1
        state = next_state
        if verbose:
            print(n_step)

#%% main code
if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    load_model, save_model, train_model, show_performance = False, False, True, True
    rl_agent = PG_Agent(n_states = env.observation_space.shape[0], n_actions = 2, \
                           lr = 0.0005, gamma = 0.99, max_n_step = 500)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/models"+"/model_lunar", compile = False)
    if train_model:
        lc, solved = learn_w(env, rl_agent, n_loop = 200, 
                             input_dim = env.observation_space.shape[0], max_n_step = rl_agent.max_n_step)
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/models"+"/model_lunar.h5")
    if train_model:
        plt.plot(smoothen(lc, 10))
    if train_model and solved:
        print("Problem solved.")
    if show_performance:
        perform(env, rl_agent, verbose = False)
    env.close()
