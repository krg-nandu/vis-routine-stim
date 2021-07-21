#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:57:40 2019

@author: heiko
Profiling
"""

import DeadLeaf as dl
import PIL
import numpy as np

imSize = np.array([50,50])
exponent = 3
distance = 10
angle = 0
abs_angle = 0
sizes = dl.default_sizes[0:15]
prob = dl.get_default_prob(exponent)[0:15]

#im = np.array(PIL.Image.open('experiment/images2/image2019_01_15_11_29_29_713129.png'))
im = dl.generate_image(exponent,0,distance,angle,abs_angle,sizes,imSize = imSize)

im = im[0]
im2 = np.mean(im,axis=-1)
im2[((im[:,:,0]==1) & (im[:,:,1]==0))]=np.nan


colors = np.unique(im2[~np.isnan(im2)])

g = dl.graph(im2,sizes,colors,prob)
rect_list = g.get_decomposition_max_explained()
