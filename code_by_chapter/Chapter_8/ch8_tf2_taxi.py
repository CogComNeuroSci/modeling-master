#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:38:02 2022

@author: tom verguts
solves the taxi problem using policy gradient (-like) algorithm
if you want an efficient algorithm for this problem instead... check out chapter 9
"""
#%% import, initialization, definitions
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ch8_tf2_pole_1 import perform
from ch8_tf2_lunar import PG_Agent
import os
import tensorflow.keras.backend as K


class PG_Agent_disc(PG_Agent):
    # PG agent with discrete state input
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
                tf.keras.layers.Dense(self.n_actions, input_shape = (self.n_states,), activation = "softmax", name = "layer")
		    	] )
        model.build()
        model.compile(optimizer = \
         tf.keras.optimizers.Adam(learning_rate = self.lr), loss = PG_loss)
        return model
    
    def learn(self, verbose: bool = True):
        state_array = np.zeros((self.max_n_step, self.n_states))
        state_array[list(range(self.max_n_step)), self.x_buffer.astype(int)] = 1
       	self.network.train_on_batch(state_array, self.y_buffer)	
        if verbose:
            print("what do you want to see?")
 
    def sample(self, state):
        state_array = np.zeros((1, self.n_states))
        state_array[0, state] = 1
        prob = np.squeeze(self.network.predict(state_array))
        action = np.random.choice(self.actions, p = prob) 
        return action

    
def learn_w(env, n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4, success_crit: int = 50):
    lc = np.zeros(n_loop)
    stop_crit = False
    loop, success = 0, 0
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
        states  = np.zeros(max_n_step)
        actions = np.zeros(max_n_step) # to construct y_buffer
        rewards = np.zeros(max_n_step) # to construct y_buffer
        for t in range(max_n_step):
            action = rl_agent.sample(state)
            next_state, reward, done, info = env.step(action)
            states[n_step]  = state
            actions[n_step] = action
            rewards[n_step] = reward
            n_step += 1
            state = next_state
            if done:
                break
        rl_agent.empty_buffer()
        rl_agent.update_buffer(n_step, states, actions, rewards)
        rl_agent.learn(verbose = False)
        lc[loop] = n_step
        loop += 1
        success += int(n_step < max_n_step)
        print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (success > success_crit)
    return lc, success > success_crit

#%% main code
if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    load_model, save_model, train_model, performance = False, False, True, False
    rl_agent = PG_Agent_disc(n_states = env.observation_space.n, n_actions = env.action_space.n, \
                           lr = 0.0005, gamma = 0.99, max_n_step = 200)
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/models/model_taxi")
    if train_model:
        lc, solved = learn_w(env, n_loop = 100, input_dim = env.observation_space.n, max_n_step = rl_agent.max_n_step)
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/models/model_taxi")
    if train_model:
        plt.plot(lc)
    if train_model and solved:
        print("Problem solved.")
    if performance:
        perform(env, rl_agent, verbose = False)
    env.close()
