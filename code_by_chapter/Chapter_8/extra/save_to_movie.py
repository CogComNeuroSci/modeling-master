#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 11:09:15 2022

@author: tom verguts
run an agent for some (n_step) steps, and then make a movie of it
"""

import os
import gym
import tensorflow as tf
from ch8_tf2_lunar import PG_Agent
from ch8_tf2_pole_2 import AgentD
import imageio
import moviepy.editor

def save_movie(imdir, movie_name, n):
    fps=1
    image_files = [os.path.join(imdir,img)
                   for img in os.listdir(imdir) if img.endswith(".png")]
    clip = moviepy.editor.ImageSequenceClip(image_files, fps = fps)
    file_name = os.path.join(os.getcwd(), "im", movie_name + "_video.mp4")
    clip.write_videofile(file_name, fps = 15)


def perform(env, rl_agent, imdir, max_n_step, verbose: bool = False):
    if not os.path.isdir(imdir):
        os.mkdir(imdir)
    state = env.reset()
    n_step, done = 0, False
    while not done:
        img = env.render(mode = "rgb_array")
        filename = imdir + "/img" + str(n_step) + ".png"
        imageio.imwrite(filename, img)
        action = rl_agent.sample(state)
        next_state, reward, done, info = env.step(action)
        n_step += 1
        state = next_state
        if verbose:
            print(n_step)
        if n_step == max_n_step:
            break

def make_png(n_step, env_type):
    # transform movie into pngs
    name = "LunarLander-v2" if env_type == "lunar" else "CartPole-v0"
    env = gym.make(name) 
    imdir = os.path.join(os.getcwd(), "im")
    if env_type == "lunar":
        rl_agent = PG_Agent(n_states = env.observation_space.shape[0], n_actions = env.action_space.n, \
                           lr = 0.001, gamma = 0.99, max_n_step = n_step)
    else:
        rl_agent = AgentD(env.observation_space.shape[0], env.action_space.n, \
                           buffer_size = 1000, epsilon_min = 0.001, epsilon_max = 0.99, \
                           epsilon_dec = 0.999, lr = 0.001, gamma = 0.99, learn_gran = 1, update_gran = 5, nhid1 = 16, nhid2 = 8)    
    file_name = os.path.join(os.getcwd(), "models", "model_" + env_type)
    rl_agent.network = tf.keras.models.load_model(file_name, compile = False)
    perform(env, rl_agent, imdir, max_n_step = n_step, verbose = False)
    env.close()
    
if __name__ == "__main__":
    n_step = 50 
    env_type = "lunar"
#    env_type = "cartpole"
    make_png(n_step = n_step, env_type = env_type)
    save_movie(imdir = os.getcwd()+"/im", movie_name = env_type, n = n_step)
