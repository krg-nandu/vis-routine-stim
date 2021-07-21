#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:53:57 2019

@author: heiko
"""


import DeadLeaf as dl
import numpy as np


sizes = np.arange(2,6,dtype=np.float)
im_size = np.array((5,5))
exponents = np.arange(0,2)

## first: exactly as in experiment
p_dist = np.load('p_dist_5.npy')
dl.save_training_data('/Users/heiko/tinytinydeadrects/training',100000,im_size=im_size,sizes=sizes,exponents=exponents,dist_probabilities=p_dist)
dl.save_training_data('/Users/heiko/tinytinydeadrects/validation',10000,im_size=im_size,sizes=sizes,exponents=exponents,dist_probabilities=p_dist)

## separate by exponent
for i in range(1,6):
    p_dist = np.load('p_dist_5_%d.npy' % i)
    dl.save_training_data('/Users/heiko/tinytinydeadrects/training%d' % i,100000,im_size=im_size,sizes=sizes,exponents=np.array([i]),dist_probabilities=p_dist)
    dl.save_training_data('/Users/heiko/tinytinydeadrects/validation%d' % i,10000,im_size=im_size,sizes=sizes,exponents=np.array([i]),dist_probabilities=p_dist)
    
