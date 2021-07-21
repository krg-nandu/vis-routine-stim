# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
      
#    
#d1 = np.load('results/H/result2018_12_11_18_59_46.npy')
#d2 = np.load('results/H/result2018_12_13_10_46_23.npy')
#d3 = np.load('results/H/result2018_12_13_18_20_01.npy')
#d4 = np.load('results/H/result2018_12_14_12_49_15.npy')
#d5 = np.load('results/H/result2018_12_14_13_26_07.npy')
#
#d = np.concatenate((d1,d2,d3,d4,d5))

#against distance
dirs = os.listdir('results')
dirs.remove('Tests')
dirs.remove('.DS_Store')

for iDir in range(len(dirs)):
    d = np.zeros((0,6))
    files = os.listdir('results'+'/'+dirs[iDir])
    for iFile in range(len(files)):
        if files[iFile][0:6]=='result':
            d = np.append(d,np.load('results'+'/'+dirs[iDir]+'/'+files[iFile]),axis=0)
    k = -1
    pTrue = np.zeros(len(np.unique(d[:,1])))
    pSame = np.zeros(len(np.unique(d[:,1])))
    for idist in np.unique(d[:,1]):
        k = k+1
        dtest = d[d[:,1]==idist,:]
        pTrue[k] = np.sum(dtest[:,4]==dtest[:,5])/dtest.shape[0]
        pSame[k] = np.sum(dtest[:,4]==1)/dtest.shape[0]
        
    pTrue = pTrue[[0,1,2,3,4,6,5,7]]
    pSame = pSame[[0,1,2,3,4,6,5,7]]
    
    pTrue = np.reshape(pTrue,(4,2))
    pSame = np.reshape(pSame,(4,2))
    
    #Sizes chosen for horizontal:
    #2 0.8897441130089403
    #5 0.7454126020552115
    #19 0.49815449458047945
    #62 0.24906879663087056
    #140 0.09888733852162014
    #Sizes chosen for diagonal:
    #1 0.8912648809369313
    #3 0.7132471428644057
    #8 0.47627923219339785
    #23 0.2460755676513367
    #56 0.0992763294998463
    pSameTrue = np.array(((0.745,0.498,0.249,0.098),(0.713,0.476,0.246,0.0992))).T
    
    blindMax = np.maximum(pSameTrue,1-pSameTrue)
    matchingMaxx = pSameTrue**2+(1-pSameTrue)**2
    
    fig = plt.figure(figsize=(10,5))
    fig.suptitle(dirs[iDir])
    
    plt.subplot(1,2,1)
    plt.plot(pSameTrue,'k--',alpha=.5)
    plt.plot(pSame)
    plt.title('P(judged same)')
    plt.ylim([0,1])
    plt.subplot(1,2,2)
    plt.plot(blindMax,'k--',alpha=.5)
    plt.plot(matchingMaxx,'k:',alpha=.5)
    plt.plot(pTrue)
    plt.title('P(judged correctly)')
    plt.ylim([0.5,1])
    
    # nColors
    k = -1
    pTrue = np.zeros(len(np.unique(d[:,0])))
    pSame = np.zeros(len(np.unique(d[:,0])))
    for inCol in np.unique(d[:,0]):
        k = k+1
        dtest = d[d[:,0]==inCol,:]
        pTrue[k] = np.sum(dtest[:,4]==dtest[:,5])/dtest.shape[0]
        pSame[k] = np.sum(dtest[:,4]==1)/dtest.shape[0]
        
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.plot(np.unique(d[:,0]),pSame,'k.-')
    plt.title('P(judged same)')
    plt.xlabel('nColors')
    plt.xticks([2,3,4,6,8])
    plt.ylim([0,1])
    
    plt.subplot(1,2,2)
    plt.plot(np.unique(d[:,0]),pTrue,'k.-')
    plt.title('P(judged correctly)')
    plt.xlabel('nColors')
    plt.xticks([2,3,4,6,8])
    plt.ylim([0.5,1])