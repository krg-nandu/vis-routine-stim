#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:32:34 2019

@author: heiko
"""

import DeadLeaf as dl
import numpy as np
import tqdm

nCols = 5
imSize= [3,3]
sizes = [1,3]
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


nSamp = 5000

l = []

for iSamp in tqdm.trange(nSamp):
    samp = [0,None]
    while samp[1] is None:
        samp = g.get_decomposition_explained_bias(points,silent=True)
    l.append(samp)
    
estimate = [s[1] for s in l]
logPPos = [s[2] for s in l]
logPVis = [s[3] for s in l]
logPCorrection = [s[4] for s in l]

lik = np.array(logPPos)-np.array(logPVis)+np.array(logPCorrection)
lik = np.exp(lik-np.max(lik))
lik = lik/np.sum(lik)

prob = np.sum(lik*np.array(estimate))

# The absolute minimal image imSize = [1,2]
# it works!
nCols = 4
sizes = [1,2]
minimal_image = np.array([[0.,0.]])
points = [[0,0],[0,1]]
g = dl.graph(minimal_image,sizes=sizes,colors=np.linspace(0,1,nCols))


nSamp = 10000

l = []

for iSamp in tqdm.trange(nSamp):
    samp = [0,None]
    while samp[1] is None:
        samp = g.get_decomposition(points,silent=True)
    l.append(samp)
    
estimate = [s[1] for s in l]
logPPos = [s[2] for s in l]
logPVis = [s[3] for s in l]
logPCorrection = [s[4] for s in l]

lik = np.array(logPPos)-np.array(logPVis)
lik = np.exp(lik-np.max(lik))
lik = lik/np.sum(lik)

prob = np.sum(lik*np.array(estimate))



# The absolute minimal image imSize = [1,2]
# biased sampling works!

import DeadLeaf as dl
import numpy as np
import tqdm


nCols = 4
sizes = [1,2]
minimal_image = np.array([[0.,0.]])
points = [[0,0],[0,1]]
g = dl.graph(minimal_image,sizes=sizes,colors=np.linspace(0,1,nCols))


nSamp = 1000

l = []

for iSamp in tqdm.trange(nSamp):
    samp = [0,None]
    while samp[1] is None:
        samp = g.get_decomposition_explained_bias(points,silent=True)
    l.append(samp)
    
estimate = [s[1] for s in l]
logPPos = [s[2] for s in l]
logPVis = [s[3] for s in l]
logPCorrection = [s[4] for s in l]

lik = np.array(logPPos)-np.array(logPVis)+np.array(logPCorrection)
lik = np.exp(lik-np.max(lik))
lik = lik/np.sum(lik)

prob = np.sum(lik*np.array(estimate))


# The absolute minimal image imSize = [1,2]
# getting the full graph
from DeadLeaf import *

nCols = 4
sizes = [1,2]
minimal_image = np.array([[0.,0.]])
points = [[0,0],[0,1]]
colors=np.linspace(0,1,nCols)
#g = dl.graph(minimal_image,sizes=sizes,colors=np.linspace(0,1,nCols))

sizes = np.reshape(np.concatenate(np.meshgrid(sizes,sizes),axis=0),[2,len(sizes)**2]).transpose()
prob = np.ones(2)
prob = np.outer(prob,prob).flatten()
prob = np.ones(len(sizes))
prob = prob * (sizes[:,0]+imSize[0]-1)/(np.max(sizes[:,0])+imSize[0]-1)
prob = prob * (sizes[:,1]+imSize[1]-1)/(np.max(sizes[:,1])+imSize[1]-1)
prob = prob/np.sum(prob)
probc = prob.cumsum()

imSize = np.array([1,2])

points = np.array(points)
n0 = node()
im = np.copy(minimal_image)
n0.add_children(im,sizes,colors,prob,silent=True)
nodes = [n0]
images = [im]
k = 0
p_node = 0
p_nodes = [p_node]
p_node_same = 0
p_same = [p_node_same]
p_prior = 1
p_priors = [p_prior]
same_rect_node = None
same_rect = [same_rect_node]

while len(nodes)>0:
    n = nodes[-1]
    im = images[-1]
    if len(nodes)==1:
        k = k+1
        print('started top-level child %d' % k)
    if len(n.children)>0:
        rect = n.children.pop()
        p_child = n.probChild.pop()
        p_invis = n.probInvisible
        same_rect_node = same_rect[-1]
        n_new = node()
        im_new = np.copy(im)
        idx_x = rect[0]
        idx_y = rect[1]
        sizx = rect[2]
        sizy = rect[3]
        im_new[int(max(idx_x,0)):int(max(0,idx_x+sizx)),int(max(idx_y,0)):int(max(0,idx_y+sizy))] = np.nan
        if same_rect_node is None:
            if np.all((idx_x<=points[:,0]) & (idx_x+sizx >points[:,0]) 
                    & (idx_y<=points[:,1]) & (idx_y+sizy >points[:,1])):
                same_rect_node = True
            elif np.any((idx_x<=points[:,0]) & (idx_x+sizx >points[:,0]) 
                    & (idx_y<=points[:,1]) & (idx_y+sizy >points[:,1])):
                same_rect_node = False
        same_rect.append(same_rect_node)
        n_new.add_children(im_new,sizes,colors,prob,silent=True)
        p_prior_new = p_priors[-1]*p_child/(1-p_invis)
        p_node_same = 0
        p_same.append(p_node_same)
        p_node = 0
        p_nodes.append(p_node)
        p_priors.append(p_prior_new)
        nodes.append(n_new)
        images.append(im_new)
    else:
        im = images.pop()
        n = nodes.pop()
        p_prior = p_priors.pop()
        p_node = p_nodes.pop()
        p_node_same = p_same.pop()
        same_rect_node = same_rect.pop()
        if len(nodes)>0:
            if np.all(np.isnan(im)):
                p_nodes[-1] = p_nodes[-1] + p_prior
                if same_rect_node:
                    p_same[-1] = p_same[-1] + p_prior
            else:
                p_nodes[-1] = p_nodes[-1] + p_node
                p_same[-1] = p_same[-1] + p_node_same






# test that [1,3] gives same result
# it does not, but it should not either?

import DeadLeaf as dl
import numpy as np
import tqdm
nCols = 4
sizes = [1,2]
minimal_image = np.array([[0.,0.,0.]])
points = [[0,0],[0,1]]
g = dl.graph(minimal_image,sizes=sizes,colors=np.linspace(0,1,nCols))


nSamp = 1000

l = []

for iSamp in tqdm.trange(nSamp):
    samp = [0,None]
    while samp[1] is None:
        samp = g.get_decomposition_explained_bias(points,silent=True)
    l.append(samp)
    
estimate = [s[1] for s in l]
logPPos = [s[2] for s in l]
logPVis = [s[3] for s in l]
logPCorrection = [s[4] for s in l]

lik = np.array(logPPos)-np.array(logPVis)+np.array(logPCorrection)
lik = np.exp(lik-np.max(lik))
lik = lik/np.sum(lik)

prob = np.sum(lik*np.array(estimate))