#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:18:00 2020

@author: tom verguts
as you can see... under construction
"""

import gym, time
env = gym.make("MsPacman-v0")

env.render()
time.sleep(1)
# do something interesting here :-)

env.close()