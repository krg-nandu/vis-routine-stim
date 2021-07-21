#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:53:57 2019

@author: heiko
"""


import DeadLeaf as dl
import numpy as np

sizes=5*np.arange(1,7)
im_size=np.array((30,30))

p_dist = np.load('p_dist_30.npy')
dl.save_training_data('/Users/heiko/tinydeadrects/training',100000,im_size=im_size,distances=None,sizes=sizes,dist_probabilities=p_dist)
dl.save_training_data('/Users/heiko/tinydeadrects/validation',10000,im_size=im_size,distances=None,sizes=sizes,dist_probabilities=p_dist)

## separate by exponent
for i in range(1,6):
    p_dist = np.load('p_dist_30_%d.npy' % i)
    dl.save_training_data('/Users/heiko/tinydeadrects/training%d' % i,100000,im_size=im_size,distances=None,sizes=sizes,exponents=np.array([i]),dist_probabilities=p_dist)
    dl.save_training_data('/Users/heiko/tinydeadrects/validation%d' % i,10000,im_size=im_size,distances=None,sizes=sizes,exponents=np.array([i]),dist_probabilities=p_dist)
    
