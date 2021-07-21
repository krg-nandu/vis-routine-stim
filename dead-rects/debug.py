#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:28:56 2019

@author: heiko
"""

import DeadLeaf as df
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time
#
#t=df.gen_rect_leaf([25,25],border=True)
#
sizes = 5*np.arange(1,80,dtype='float')
prob = (sizes/np.min(sizes)) **-1.5
imSize = np.array([400,300])
colors = np.linspace(0,1,9)

t=df.gen_rect_leaf(imSize,prob=prob,sizes=sizes,border=False,colors = colors)
#plt.figure(figsize=(25,25))
#plt.imshow(t[0])
im = t[0]

def fast_rect_conv(im,rect_size):
    # always convolves along the first axis
    imOut = np.zeros(np.array(im.shape)+(rect_size-1,0))
    current = np.zeros(im.shape[1])
    for iC in range(im.shape[0]+rect_size-1):
        if iC<im.shape[0]:
            current = current+im[iC]
        if iC>=rect_size:
            current = current-im[iC-rect_size]
        imOut[iC] = current
    return imOut


#plt.figure(figsize=(25,25))

imT = (im-0.5)**2

iSize = [30,20]

t0 = time.time()
recty = np.ones((iSize[0],1))
rectx = np.ones((1,iSize[1]))

imTest = signal.convolve2d(imT,rectx,'full')
imTest = signal.convolve2d(imTest,recty,'full')

t1 = time.time()
imTest2 = fast_rect_conv(imT,iSize[0])
imTest2 = fast_rect_conv(imTest2.transpose(),iSize[1]).transpose()
t2 = time.time()