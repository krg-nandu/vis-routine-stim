#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:58:28 2018

@author: Heiko
"""


#from psychopy import visual,core,event,monitors
import numpy as np
import PIL
import pickle
import os
import tqdm

# This is the only way I found to do this. It looks like a HORRIBLE IDEA
# but unfortunately it works so far.
import sys
sys.path.append('..')
import DeadLeaf as dl
sys.path.pop(-1)

import datetime

im_folder = './imagesFrozen/'

exponents = np.arange(5)+1
    
sizes = 5*np.arange(1,80,dtype='float')
imSize = np.array([300,300])
factorSize = 3

distances = [5,10,20,40,80]
distancesd = [4,7,14,28,57]
    


num_colors= 9;


#os.mkdir(im_folder)
 

xlen = 2
#distance = results[i,1]
#n_c = int(results[i,0])
#angle = results[i,2]
#abs_angle = results[i,3]

Nimages = 25
t = tqdm.tqdm(total=Nimages*len(exponents)*2*len(distances)*2,position=0,smoothing=0)
counter = 0

for i in range(Nimages):
    for exponent in tqdm.tqdm(exponents,position=1):
        for iangle in tqdm.trange(2,position=2):
            if iangle:
                dist = distancesd
            else:
                dist = distances 
            for idist in tqdm.tqdm(dist,position=3):
                for iabs_angle in range(2):
                    image,rects,positions_im,same,col = dl.generate_image(
                            exponent,0,idist,iangle,iabs_angle,
                            imSize=imSize,sizes=sizes,num_colors=num_colors)
                    # check internal consistency
                    same_calculated = dl.test_positions(rects,positions_im)
                    if same_calculated != same:
                        raise ValueError('Same calculation is internally inconsistent!')
                    for ix in range(1,xlen+1):
                        for ip in range(2):
                            image[np.uint(positions_im[ip,0]+ix),np.uint(positions_im[ip,1]+ix)] = [1,0,0]
                            image[np.uint(positions_im[ip,0]+ix),np.uint(positions_im[ip,1]-ix)] = [1,0,0]
                            image[np.uint(positions_im[ip,0]-ix),np.uint(positions_im[ip,1]+ix)] = [1,0,0]
                            image[np.uint(positions_im[ip,0]-ix),np.uint(positions_im[ip,1]-ix)] = [1,0,0]
                    im_name = im_folder+"image%d_%d_%d_%d_%d_%d_%d.png" % (exponent,num_colors,idist,iangle,iabs_angle,i,0)
                    im_pil = PIL.Image.fromarray(np.uint8(255*image))
                    im_pil.save(im_name,'PNG')
                    rect_name = im_folder+"rect%d_%d_%d_%d_%d_%d_%d.npy" % (exponent,num_colors,idist,iangle,iabs_angle,i,0)
                    np.save(rect_name,rects)
                    same_name = im_folder+"same%d_%d_%d_%d_%d_%d_%d.npy" % (exponent,num_colors,idist,iangle,iabs_angle,i,0)
                    np.save(same_name,np.array([same,col,positions_im[0,0],positions_im[0,1],positions_im[1,0],positions_im[1,1]]))
                    image = dl.generate_image_from_rects(imSize,rects,border=True,colors=np.linspace(0,1,num_colors))
                    for ix in range(1,xlen+1):
                        for ip in range(2):
                            image[np.uint(positions_im[ip,0]+ix),np.uint(positions_im[ip,1]+ix)] = [1,0,0]
                            image[np.uint(positions_im[ip,0]+ix),np.uint(positions_im[ip,1]-ix)] = [1,0,0]
                            image[np.uint(positions_im[ip,0]-ix),np.uint(positions_im[ip,1]+ix)] = [1,0,0]
                            image[np.uint(positions_im[ip,0]-ix),np.uint(positions_im[ip,1]-ix)] = [1,0,0]
                    im_name = im_folder+"image%d_%d_%d_%d_%d_%d_%d.png" % (exponent,num_colors,idist,iangle,iabs_angle,i,1)
                    im_pil = PIL.Image.fromarray(np.uint8(255*image))
                    im_pil.save(im_name,'PNG')
                    rect_name = im_folder+"rect%d_%d_%d_%d_%d_%d_%d.npy" % (exponent,num_colors,idist,iangle,iabs_angle,i,1)
                    np.save(rect_name,rects)
                    same_name = im_folder+"same%d_%d_%d_%d_%d_%d_%d.npy" % (exponent,num_colors,idist,iangle,iabs_angle,i,1)
                    np.save(same_name,np.array([same,col,positions_im[0,0],positions_im[0,1],positions_im[1,0],positions_im[1,1]]))
                    t.update(1)
t.close()                       