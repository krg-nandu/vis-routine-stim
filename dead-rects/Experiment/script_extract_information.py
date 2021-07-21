#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:42:17 2019

@author: heiko
"""

import numpy as np
import os
import tqdm
import time
import PIL
import scipy.ndimage as ndimage


# This is the only way I found to do this. It looks like a HORRIBLE IDEA
# but unfortunately it works so far.
import sys
sys.path.append('..')
import DeadLeaf as dl
sys.path.pop(-1)

imagesFolder = 'imagesFrozen'

files = os.listdir(imagesFolder)
rectFiles = [f for f in files if f[0:4]=='rect']
rectFiles.sort()
imageFiles = [f for f in files if f[0:4]=='imag']
imageFiles.sort()

imSize = np.array([300,300]);

# extract the list of visible rectangles
# two versions-> first remove all rectangles which are invisible
#                second remove rectangles which are not distinguishable from BG

# the slow way...
#class rect:
#    def __init__(self,entry):
#        self.entry = entry
#        self.x = entry[0]
#        self.y = entry[1]
#        self.sizex = entry[2]
#        self.sizey = entry[3]
#        self.c     = entry[4]
#        self.visible = np.zeros((self.sizex,self.sizey),dtype=np.bool)
#        xrelmin = min(max(0-self.x,0),self.sizex)
#        yrelmin = min(max(0-self.y,0),self.sizey)
#        xrelmax = min(max(imSize[0]-self.x,0),self.sizex)
#        yrelmax = min(max(imSize[1]-self.y,0),self.sizey)
#        self.visible[xrelmin:xrelmax,yrelmin:yrelmax] = True
#    def hide(self,entry):
#        xrelmin = entry[0]-self.x
#        yrelmin = entry[1]-self.y
#        xrelmax = entry[0]+entry[2]-self.x
#        yrelmax = entry[1]+entry[3]-self.y
#        if xrelmin > self.sizex:
#            xrelmin = self.sizex
#            xrelmax = self.sizex
#        elif xrelmax > self.sizex:
#            xrelmax = self.sizex
#        if yrelmin > self.sizey:
#            yrelmin = self.sizey
#            yrelmax = self.sizey
#        elif yrelmax > self.sizey:
#            yrelmax = self.sizey
#        if xrelmax < 0:
#            xrelmax = 0
#            xrelmin = 0
#        elif xrelmin < 0:
#            xrelmin = 0
#        if yrelmax < 0:
#            yrelmax = 0
#            yrelmin = 0
#        elif yrelmin < 0:
#            yrelmin = 0
#        #xrelmin = min(max(entry[0]-self.x,0),self.sizex)
#        #yrelmin = min(max(entry[1]-self.y,0),self.sizey)
#        #xrelmax = min(max(entry[0]+entry[2]-self.x,0),self.sizex)
#        #yrelmax = min(max(entry[1]+entry[3]-self.y,0),self.sizey)
#        self.visible[xrelmin:xrelmax,yrelmin:yrelmax] = False
#        
#        
#        
#for iFile in tqdm.trange(len(rectFiles),position=0):
##for iFile in tqdm.trange(2,position=0):
#    rectsOriginal = np.load(imagesFolder+'/'+rectFiles[iFile])
#    
#    image = -np.ones(imSize)
#    rectList1 = []
#    rectList2 = []
#    for iRect in tqdm.trange(rectsOriginal.shape[0],position=1):
#        entry = rectsOriginal[len(rectsOriginal)-iRect-1]
#        if np.any(image[int(max(entry[0],0)):int(max(0,entry[0]+entry[2])),
#              int(max(entry[1],0)):int(max(0,entry[1]+entry[3]))]!=entry[4]):
#            inList2=True
#        else:
#            inList2=False
#        for iR in range(len(rectList1)):
#            rectList1[iR].hide(entry)
#        if inList2:
#            for iR in range(len(rectList2)):
#                rectList2[iR].hide(entry)
#        rectList1.append(rect(entry))  
#        if inList2:
#            rectList2.append(rect(entry))   
#        if not (iRect % 10):
#            rectList1 = [r for r in rectList1 if np.any(r.visible)]
#            rectList2 = [r for r in rectList2 if np.any(r.visible)]
#        image[int(max(entry[0],0)):int(max(0,entry[0]+entry[2])),
#              int(max(entry[1],0)):int(max(0,entry[1]+entry[3]))] = entry[4]
#    rectList1 = [r for r in rectList1 if np.any(r.visible)]
#    rectList2 = [r for r in rectList2 if np.any(r.visible)]
#    rectsNew = np.array([rect.entry for rect in rectList1])
#    rectsNew2 = np.array([rect.entry for rect in rectList2])
#    np.save(imagesFolder+'/visible'+rectFiles[iFile],rectsNew)
#    np.save(imagesFolder+'/visible2'+rectFiles[iFile],rectsNew2)

# much faster & thus possible way to do this:
for iFile in tqdm.trange(len(rectFiles),position=0):
    rectsOriginal = np.load(imagesFolder+'/'+rectFiles[iFile])
    image = -np.ones(imSize)
    rectList1 = list([])
    for iRect in range(rectsOriginal.shape[0]):
        entry = rectsOriginal[iRect]
        if np.any(image[int(max(entry[0],0)):int(max(0,entry[0]+entry[2])),
                        int(max(entry[1],0)):int(max(0,entry[1]+entry[3]))]==-1):
            rectList1.append(entry)
            image[int(max(entry[0],0)):int(max(0,entry[0]+entry[2])),
                  int(max(entry[1],0)):int(max(0,entry[1]+entry[3]))] = entry[4]
    rectsNew = np.array(rectList1)
    np.save(imagesFolder+'/visible'+rectFiles[iFile],rectsNew)
    image = -np.ones(imSize)
    rectList2 = list([])
    for iRect in range(rectsNew.shape[0]):
        entry = rectsNew[rectsNew.shape[0]-iRect-1]
        if np.any(image[int(max(entry[0],0)):int(max(0,entry[0]+entry[2])),
                        int(max(entry[1],0)):int(max(0,entry[1]+entry[3]))]!=entry[4]):
            rectList2.append(entry)
            image[int(max(entry[0],0)):int(max(0,entry[0]+entry[2])),
                  int(max(entry[1],0)):int(max(0,entry[1]+entry[3]))] = entry[4]
    rectList2.reverse()
    rectsNew2 = np.array(rectList2)
    np.save(imagesFolder+'/visible2'+rectFiles[iFile],rectsNew2)
    same = np.load(imagesFolder+'/same'+rectFiles[iFile][4:])
    
    fname = rectFiles[iFile].split('_')
    distance = int(fname[2])
    angle = int(fname[3])
    abs_angle = int(fname[4])
    if angle and not abs_angle:
        pos = [[-distance/2,-distance/2],[distance/2,distance/2]]
    elif angle and abs_angle:
        pos = [[-distance/2,distance/2],[distance/2,-distance/2]]
    elif not angle and not abs_angle:
        pos = [[-distance/2,0],[distance/2,0]]
    elif not angle and abs_angle: 
        pos = [[0,-distance/2],[0,distance/2]]
    pos = np.floor(np.array(pos))
    
    positions = pos
    positions = np.floor(positions)
    positions_im = np.zeros_like(positions)
    positions_im[:,1] = np.floor(imSize/2)+positions[:,0]
    positions_im[:,0] = np.floor(imSize/2)-positions[:,1]-1
    
    sameOriginal = dl.test_positions(rectsOriginal,positions_im)
    same1 = dl.test_positions(rectsNew,positions_im)
    if sameOriginal != same1:
        print('SOMETHING WENT WRONG!\n Processing changed truth!')
    if sameOriginal != same[0]:
        print('SOMETHING WENT WRONG!\n Saved result is not the one calculated now!')
    same = np.concatenate((np.array([same[0],same[1],dl.test_positions(rectsNew2,positions_im)]),same[2:]))
    same = np.save(imagesFolder+'/same'+rectFiles[iFile][4:],same)
    

### Actually extracting information from images!
statistics = np.zeros((len(imageFiles),25))

for iFile in tqdm.trange(len(imageFiles),position=0):
    image = PIL.Image.open(imagesFolder+'/'+imageFiles[iFile])
    fname = imageFiles[iFile].split('_')
    n_c = int(fname[1])
    distance = int(fname[2])
    angle = int(fname[3])
    abs_angle = int(fname[4])
    imageNumber = int(fname[5])
    border = int(fname[6][0])
    exponent = int(fname[0][-1])
    
    same_name = imagesFolder+ '/' + "same%d_%d_%d_%d_%d_%d_%d.npy" % (exponent,n_c,distance,angle,abs_angle,imageNumber,0)               
    same = np.load(same_name)
    colPositions = int(same[1]/8*255)
    
    if angle and not abs_angle:
        pos = [[-distance/2,-distance/2],[distance/2,distance/2]]
    elif angle and abs_angle:
        pos = [[-distance/2,distance/2],[distance/2,-distance/2]]
    elif not angle and not abs_angle:
        pos = [[-distance/2,0],[distance/2,0]]
    elif not angle and abs_angle: 
        pos = [[0,-distance/2],[0,distance/2]]
    pos = np.floor(np.array(pos))
    
    positions = pos
    positions = np.floor(positions)
    positions_im = np.zeros_like(positions)
    positions_im[:,1] = np.floor(imSize/2)+positions[:,0]
    positions_im[:,0] = np.floor(imSize/2)-positions[:,1]-1
    
    imArray = np.array(image)
    ## extract line connecting the two points and the two connections along the cardinals
    if positions_im[0,0]>positions_im[1,0]:
        lineX = range(int(positions_im[0,0]),int(positions_im[1,0]-1),-1)
    else:
        lineX = range(int(positions_im[0,0]),int(positions_im[1,0]+1))
    if positions_im[0,1]>positions_im[1,1]:
        lineY = range(int(positions_im[0,1]),int(positions_im[1,1]-1),-1)
    else:
        lineY = range(int(positions_im[0,1]),int(positions_im[1,1]+1))
    line = imArray[lineX,lineY,:]
    lineNumOtherColor = np.sum(np.any(line != colPositions,axis=1))
    lineUniqueColors = len(np.unique(line,axis=0))
    statistics[iFile,0] = exponent
    statistics[iFile,1] = n_c
    statistics[iFile,2] = distance
    statistics[iFile,3] = angle
    statistics[iFile,4] = abs_angle
    statistics[iFile,5] = imageNumber
    statistics[iFile,6] = same[0]
    statistics[iFile,7] = lineNumOtherColor
    statistics[iFile,8] = lineUniqueColors