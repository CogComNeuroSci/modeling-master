#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 11:09:15 2022

@author: tom verguts
function save_movie doesn't work.. suggestions welcome!
"""

import os
import gym
import tensorflow as tf
from ch8_tf2_lunar import PG_Agent
import imageio
import moviepy.editor

def save_movie(imdir, movie_name, n):
    fps=1
    image_files = [os.path.join(imdir,img)
                   for img in os.listdir(imdir) if img.endswith(".png")]
    print(image_files)
    clip = moviepy.editor.ImageSequenceClip(image_files, fps = fps)
    clip.write_video(os.getcwd() + movie_name + "_video.mp4", fps = 15)


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

def make_png(n_step):
    # transform movie into pngs
    env = gym.make('LunarLander-v2')
    imdir = os.getcwd() + "/im"
    rl_agent = PG_Agent(n_states = env.observation_space.shape[0], n_actions = env.action_space.n, \
                           lr = 0.001, gamma = 0.99, max_n_step = n_step)
    rl_agent.network = tf.keras.models.load_model(os.getcwd()+"/model_lunar", compile = False)
    perform(env, rl_agent, imdir, max_n_step = n_step, verbose = False)
    env.close()
    
if __name__ == "__main__":
    n_step = 500 
    make_png(n_step = n_step)
    #save_movie(imdir = os.getcwd()+"/im", movie_name = "lunar", n = n_step)
