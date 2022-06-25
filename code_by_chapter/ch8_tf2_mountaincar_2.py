#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:43:48 2022

@author: tom verguts
the mountain car problem: works but not amazingly efficient
here i try using subgoals and some other stuff
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from ch8_tf2_pole_1 import perform
from ch8_tf2_pole_2 import AgentD


class AgentDP(AgentD): # double Q and prioritized replay
    def __init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, 
                 epsilon_dec, lr, gamma, learn_gran, update_gran, nhid1, nhid2):
        AgentD.__init__(self, n_states, n_actions, buffer_size, epsilon_min, epsilon_max, 
                 epsilon_dec, lr, gamma, learn_gran, update_gran, nhid1, nhid2)

    def learn(self, n: int, verbose: bool = True): # this method is overwritten from AgentD
        #self.epsilon = self.epsilon_max # in case you want to reset epsilon on each episode
        sample_size = np.minimum(100, n)
        probs = np.exp(self.r_buffer) / np.sum(np.exp(self.r_buffer))
        print(probs)
        print(np.squeeze(probs).shape)
        sample = np.random.choice(n, sample_size, p = np.squeeze(probs)) # prioritize feedback
        q = self.network.predict(self.x_buffer[sample])
        q_next = self.network.predict(self.xn_buffer[sample])
        q_next_target = self.network_target.predict(self.xn_buffer[sample])
        index_max = np.argmax(q_next, axis = 1) # what would the online network choose?
        q_max = q_next_target[list(range(q.shape[0])), index_max] # what does the target network think of this?
        include_v = 1 - self.d_buffer[sample]
        if verbose:
            print("x buffer:", self.x_buffer)
            print("n: ", n)
            print("q_predict", q)
        q_target = q.copy() 
        target_indices = np.dot(self.y_buffer[sample], np.arange(self.n_actions)).astype(int)
        q_target[list(range(q.shape[0])), target_indices] = np.squeeze(self.r_buffer[sample])
        q_target[list(range(q.shape[0])), target_indices] += self.gamma*q_max * np.squeeze(include_v)
       	self.network.fit(self.x_buffer[sample], q_target, batch_size = 64, epochs = 2000, verbose = 0)	
        if verbose:
            print("q_target", q_target)

def learn_w(n_loop: int = 100, max_n_step: int = 200, input_dim: int = 4):
    lc = np.zeros(n_loop)
    buffer_count = 0
    stop_crit = False
    loop, success = 0, 0
    best_location = -0.4
    # learn
    while not stop_crit:
        print("episode loop", loop)
        n_step, done = 0, False
        state = env.reset()
        data = np.zeros((max_n_step, input_dim*2 + 3)) # data for this loop
        while not done:
            action = rl_agent.sample(state)
            next_state, reward, done, info = env.step(action)
            data[n_step, 0:input_dim] = state
            data[n_step, input_dim:2*input_dim] = next_state
            data[n_step, -3]  = action
#            if loop==0:
#                local_score = 200
#            else:
#                local_score = np.mean(lc[np.maximum(loop-5,0):loop])
#            if local_score >= 99.5:
#                fb = reward + (state[0]>0.1)*50
#                if n_step < 200-1:
#                    done = int(state[0]>0.1)
#            else:
            #fb = reward + (next_state[0] > best_location)*5 + int(done)*(n_step<(200-1))*10
            fb = reward + (next_state[0]>0.3)*10
#            fb = reward + int(done)*(n_step<(200-1))*50
            data[n_step, -2]  = fb
            data[n_step, -1]  = done
            n_step += 1
            state = next_state
            best_location = np.maximum(state[0], best_location)
        print(state)
        print(best_location)
#       print(local_score, state[0])    
#        print(fb)    
        buffer_count = rl_agent.update_buffer(data, n_step, buffer_count)
        if not loop % rl_agent.update_gran:
            rl_agent.update_q()
        if (not loop % rl_agent.learn_gran) and (buffer_count > 500): # don't learn first 500 trials
            rl_agent.learn(buffer_count, verbose = False)
        lc[loop] = n_step
        loop += 1
        success += (fb > 0)
        print("n steps = " + str(n_step) + "\n")
        stop_crit = (loop == n_loop) or (success > 10)

    return lc, success > 10

        
if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    load_model, save_model, train_model = False, False, True
    rl_agent = AgentDP(env.observation_space.shape[0], env.action_space.n, \
                           buffer_size = 1000, epsilon_min = 0.01, epsilon_max = 1, \
                           epsilon_dec = 0.9999, lr = 0.001, gamma = 0.999, learn_gran = 1, update_gran = 5, nhid1 = 16, nhid2 = 16) # update_gran = 1, 
    if load_model:
        rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/model_mountaincar")
    if train_model:
        lc, solved = learn_w(n_loop = 100, max_n_step = 200, input_dim = env.observation_space.shape[0])
    if save_model:
        tf.keras.models.save_model(rl_agent.network, os.getcwd()+"/model_mountaincar")
    if train_model:
        plt.plot(lc)
    if train_model and solved:
        print("Problem solved.")
    perform(env, rl_agent, verbose = False)
    env.close()