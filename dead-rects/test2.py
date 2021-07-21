#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:04:13 2018

@author: heiko
"""
import numpy as np
import DeadLeaf as dl

sizes = 5*np.arange(1,20,dtype='float')
prob = (sizes/np.min(sizes)) **-1.5
imSize = np.array([50,50])
im = dl.gen_rect_leaf(imSize,sizes,prob=prob)
imT = im[0]

g = dl.graph(imT,sizes,[0,.5,1],prob)
t = g.get_decomposition_explained_bias()