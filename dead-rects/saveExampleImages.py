# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import DeadLeaf as dl
import PIL

sizes = 5*np.array([np.arange(1,150,dtype='float'),np.arange(1,150,dtype='float')])
prob = (sizes[0]/np.min(sizes[0])) **-3
sizes = sizes.transpose()
im = dl.gen_rect_leaf([800,800],
          sizes=sizes,
          prob = prob,
          grid=1,
          colors=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('sizeInvariantSquares.png','PNG')

sizes = 5*np.array([np.arange(1,150,dtype='float'),np.arange(1,150,dtype='float')])
prob = (sizes[0]/np.min(sizes[0])) **-3
sizes = sizes.transpose()
im = dl.gen_rect_leaf([800,800],
          sizes=sizes,
          prob = prob,
          grid=5,
          colors=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('sizeInvariantSquaresGrid.png','PNG')

sizes = 5*np.arange(1,150,dtype='float')
prob = (sizes/np.min(sizes)) **-1.5
im = dl.gen_rect_leaf([800,800],
          sizes=sizes,
          prob = prob,
          grid=1,
          colors=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('sizeInvariantRect.png','PNG')

sizes = 5*np.arange(1,150,dtype='float')
prob = (sizes/np.min(sizes)) **-1.5
im = dl.gen_rect_leaf([800,800],
          sizes=sizes,
          prob = prob,
          grid=5,
          colors=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('sizeInvariantRectGrid.png','PNG')

im = dl.gen_rect_leaf([400,400],
          sizes=np.array([[5,20],[20,5],[10,20],[20,10],[20,20],[10,10]]),
          grid=5,
          colors=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('manyColors.png')


im = dl.gen_rect_leaf([100,100],
          sizes=np.array([[5,10],[5,20],[20,5],[10,20],[20,10],[20,20],[10,5],[10,10],[5,5]]),
          grid=5,noise=.2,
          colors=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('manyColorsNoise.png')

im = dl.gen_rect_leaf([100,100],
          sizes=[[5,10],[5,20],[20,5],[10,20],[20,10],[20,20],[10,5],[10,10],[5,5],
                 [40,40],[40,20],[20,40],[80,40],[40,80]],
          grid=5,noise=.2,noiseType = 'uniform',
          colors=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('manyColorsNoiseU.png')

im = dl.gen_rect_leaf([100,100],
          sizes=[[5,10],[5,20],[20,5],[10,20],[20,10],[20,20],[10,5],[10,10],[5,5],
                 [40,40],[40,20],[20,40],[80,40],[40,80]],
          grid=5,noise=0,
          colors=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('manyColorsLarge.png')


im = dl.gen_rect_leaf([100,100],
          sizes=[[5,10],[5,20],[20,5],[10,20],[20,10],[20,20],[10,5],[10,10],[5,5]],
          grid=5,
          colors=[0,0.5,1])
im2 = PIL.Image.fromarray(255*im[0])
im3 = im2.convert('L')
im3.save('fewColors.png')