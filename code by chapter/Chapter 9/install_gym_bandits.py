#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:12:36 2020

@author: tom
"""

import setuptools
from setuptools import setup

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setup(name='gym_bandits',
      version='0.0.1',
      author='Per Mattsson',
      author_email='magni84@gmail.com',
      description='Implements multi-armed bandits',
      url='https://github.com/magni84/gym_bandits',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
      install_requires=['gym', 'numpy']
)
