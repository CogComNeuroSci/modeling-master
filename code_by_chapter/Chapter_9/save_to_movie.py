#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 11:09:15 2022

@author: tom verguts
make a movie of sequence of png files
"""

import os
import moviepy.editor

def save_movie(imdir, movie_name, n):
    fps=1
    image_files = [os.path.join(imdir,img)
                   for img in os.listdir(imdir) if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.editor.ImageSequenceClip(image_files, fps = fps)
    file_name = os.path.join(os.getcwd(), "im", movie_name + "_video.mp4")
    clip.write_videofile(file_name, fps = 15)


