#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:31:17 2019

@author: heiko
"""


import DeadLeaf as dl
import numpy as np
import tqdm

nCols = 5
imSize= [2,2]
sizes = [1,2]
exponent = 3
distance = None
angle = 0
abs_angle = 0

test_image = dl.generate_image(exponent,0,sizes,distance,angle,abs_angle,imSize=np.array(imSize),num_colors=nCols,mark_points=True)
points = test_image[2]
same_rect = test_image[3]
rects = test_image[1]
im = test_image[0]

g = dl.graph(im[:,:,2],sizes=sizes,colors=np.linspace(0,1,nCols),prob=dl.get_default_prob(exponent,sizes))

p_node_same, p_node = g.get_exact_prob(points)
print('\n')
print(p_node_same)
print(p_node)
print(p_node_same/p_node)
print(same_rect)