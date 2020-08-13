#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:47:51 2020

@author: tom
"""

from PIL import Image
img = Image.open('contour.png').convert('LA')
img.save('contour_gray.png')