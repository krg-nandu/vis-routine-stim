# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
   

folder= 'resultsFrozen'
distances = [5,10,20,40,80]
distancesd = [4,7,14,28,57]

#against distance
dirs = os.listdir(folder)
#dirs.remove('test')
dirs.remove('.DS_Store')
dirs.remove('Icon\r')
k =0
while k<len(dirs):
    if 'train' in dirs[k]:
        dirs.remove(dirs[k])
    elif 'test' in dirs[k]:
        dirs.remove(dirs[k])
    else:
        k = k + 1

details=False

for iDir in range(len(dirs)):
    d = np.zeros((0,9))
    files = os.listdir(folder+'/'+dirs[iDir])
    for iFile in range(len(files)):
        if files[iFile][0:6]=='result':
            d = np.append(d,np.load(folder+'/'+dirs[iDir]+'/'+files[iFile]),axis=0)
    k = -1
    pTrue = np.zeros((len(np.unique(d[:,2])),len(np.unique(d[:,0]))))
    pSame = np.zeros((len(np.unique(d[:,2])),len(np.unique(d[:,0]))))
    pSameTrue = np.zeros((len(np.unique(d[:,2])),len(np.unique(d[:,0]))))
    for idist in np.unique(d[:,2]):
        k = k+1
        l=-1
        for iExp in np.unique(d[:,0]):
            l=l+1
            dtest = d[(d[:,2]==idist) & (d[:,0]==iExp),:]
            pTrue[k,l] = np.sum(dtest[:,6]==dtest[:,7])/dtest.shape[0]
            pSame[k,l] = np.sum(dtest[:,7]==1)/dtest.shape[0]
            pSameTrue[k,l] = np.sum(dtest[:,6]==1)/dtest.shape[0]
        
    #pTrue = pTrue[[0,1,2,3,4,5,6,7,8,9]]
    #pSame = pSame[[0,1,2,3,4,5,6,7,8,9]]
    #pSameTrue = pSameTrue[[0,1,2,3,4,5,6,7,8,9]]
    pTrue = np.reshape(pTrue,(5,2,5))
    pSame = np.reshape(pSame,(5,2,5))
    pSameTrue = np.reshape(pSameTrue,(5,2,5))
    pTrue = np.flip(pTrue,axis=1)
    pSame = np.flip(pSame,axis=1)
    pSameTrue = np.flip(pSameTrue,axis=1)
    blindMax = np.maximum(pSameTrue,1-pSameTrue)
    matchingMaxx = pSameTrue**2+(1-pSameTrue)**2
    if details:
        for iExp in range(5):
            fig = plt.figure(figsize=(15,5))
            fig.suptitle(dirs[iDir]+':Exponent=%d'%(iExp+1))
            
            plt.subplot(1,3,1)
            plt.plot(distances,np.squeeze(pSameTrue[:,0,iExp]),'k--',alpha=.5)
            plt.plot(distancesd,np.squeeze(pSameTrue[:,1,iExp]),'k--',alpha=.5)
            plt.plot(distances,np.squeeze(pSame[:,0,iExp]))
            plt.plot(distancesd,np.squeeze(pSame[:,1,iExp]))
            plt.ylabel('P(judged same)')
            plt.xlabel('distance')
            plt.ylim([0,1])
            plt.subplot(1,3,2)
            plt.plot(distances,np.squeeze(blindMax[:,0,iExp]),'k--',alpha=.5)
            plt.plot(distances,np.squeeze(matchingMaxx[:,0,iExp]),'k:',alpha=.5)
            plt.plot(distances,np.squeeze(pTrue[:,0,iExp]))
            plt.ylabel('P(judged correctly)')
            plt.xlabel('distance')
            plt.ylim([0.5,1])
            plt.subplot(1,3,3)
            plt.plot(distancesd,np.squeeze(blindMax[:,1,iExp]),'k--',alpha=.5)
            plt.plot(distancesd,np.squeeze(matchingMaxx[:,1,iExp]),'k:',alpha=.5)
            plt.plot(distancesd,np.squeeze(pTrue[:,1,iExp]))
            plt.ylim([0.5,1])
            plt.ylabel('P(judged correctly)')
            plt.xlabel('distance')
    
    fig = plt.figure(figsize=(15,5))
    fig.suptitle(dirs[iDir])
    
    plt.subplot(1,3,1)
    plt.plot(distances,np.squeeze(np.mean(pSameTrue[:,0,:],axis=1)),'k--',alpha=.5)
    plt.plot(distancesd,np.squeeze(np.mean(pSameTrue[:,1,:],axis=1)),'k--',alpha=.5)
    plt.plot(distances,np.squeeze(np.mean(pSame[:,0,:],axis=1)))
    plt.plot(distancesd,np.squeeze(np.mean(pSame[:,1,:],axis=1)))
    plt.ylabel('P(judged same)')
    plt.ylim([0,1])
    plt.subplot(1,3,2)
    plt.plot(distances,np.squeeze(np.mean(blindMax[:,0,:],axis=1)),'k--',alpha=.5)
    plt.plot(distances,np.squeeze(np.mean(matchingMaxx[:,0,:],axis=1)),'k:',alpha=.5)
    plt.plot(distances,np.squeeze(np.mean(pTrue[:,0,:],axis=1)))
    plt.ylabel('P(judged correctly)')
    plt.ylim([0.5,1])
    plt.subplot(1,3,3)
    plt.plot(distancesd,np.squeeze(np.mean(blindMax[:,1,:],axis=1)),'k--',alpha=.5)
    plt.plot(distancesd,np.squeeze(np.mean(matchingMaxx[:,1,:],axis=1)),'k:',alpha=.5)
    plt.plot(distancesd,np.squeeze(np.mean(pTrue[:,1,:],axis=1)))
    plt.ylabel('P(judged correctly)')
    plt.ylim([0.5,1])
    
    # nColors
    k = -1
    pTrue = np.zeros(len(np.unique(d[:,0])))
    pSame = np.zeros(len(np.unique(d[:,0])))
    pSameTrue = np.zeros(len(np.unique(d[:,0])))
    for exponent in np.unique(d[:,0]):
        k = k+1
        dtest = d[d[:,0]==exponent,:]
        pTrue[k] = np.sum(dtest[:,6]==dtest[:,7])/dtest.shape[0]
        pSame[k] = np.sum(dtest[:,7]==1)/dtest.shape[0]
        pSameTrue[k] = np.sum(dtest[:,6]==1)/dtest.shape[0]
        
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.plot(np.unique(d[:,0]),pSame,'k.-')
    plt.plot(np.unique(d[:,0]),pSameTrue,'k.--',alpha=.5)
    plt.title('P(judged same)')
    plt.xlabel('exponent')
    plt.xticks([1,3,5])
    plt.ylim([0,1])
    
    plt.subplot(1,2,2)
    plt.plot(np.unique(d[:,0]),pTrue,'k.-')
    plt.title('P(judged correctly)')
    plt.xlabel('exponent')
    plt.xticks([1,3,5])
    plt.ylim([0.5,1])