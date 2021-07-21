#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:01:00 2020

@author: heiko
"""

import numpy as np
from scipy.ndimage import convolve1d as conv
import matplotlib.pyplot as plt
import scipy
import time

eps = np.finfo(float).eps


def segment(compatibility, start, scale=1):
    compatibility = compatibility * scale
    segmentation = 0.5 * np.ones((compatibility.shape[1],
                                  compatibility.shape[2]))
    segmentation[start[0], start[1]] = 1
    for k in range(5):
        for i in range(compatibility.shape[1]):
            for j in range(compatibility.shape[2]):
                if not (i == start[0] and j == start[1]):
                    p = np.array([0., 0])
                    if i < (compatibility.shape[1] - 1):
                        p0 = compatibility[0, i, j] \
                            * (segmentation[i + 1, j] * np.array([1, 0])
                               + (1 - segmentation[i + 1, j])
                               * np.array([-1, 0]))
                        p += p0
                    if i > 0:
                        p1 = compatibility[1, i, j] \
                            * (segmentation[i - 1, j] * np.array([1, 0])
                               + (1-segmentation[i - 1, j])
                               * np.array([-1, 0]))
                        p += p1
                    if j < (compatibility.shape[2]-1):
                        p2 = compatibility[2, i, j] \
                            * (segmentation[i, j + 1] * np.array([1, 0])
                               + (1 - segmentation[i, j + 1])
                               * np.array([0, 1]))
                        p += p2
                    if j > 0:
                        p3 = compatibility[3, i, j] \
                            * (segmentation[i, j - 1] * np.array([1, 0])
                               + (1-segmentation[i, j - 1]) * np.array([0, 1]))
                        p += p3
                    segmentation[i, j] = np.exp(p[0]) / np.sum(np.exp(p))
        plt.imshow(segmentation)
        plt.colorbar()
        plt.show()
        for i in range(compatibility.shape[1] - 1, -1, -1):
            for j in range(compatibility.shape[2] - 1, -1, -1):
                if not (i == start[0] and j == start[1]):
                    p = np.array([0., 0.])
                    if i < (compatibility.shape[1]-1):
                        p0 = compatibility[0, i, j] \
                            * (segmentation[i + 1, j] * np.array([1, 0])
                               + (1 - segmentation[i + 1, j])
                               * np.array([-1, 0]))
                        p += p0
                    if i > 0:
                        p1 = compatibility[1, i, j] \
                            * (segmentation[i - 1, j] * np.array([1, 0])
                               + (1 - segmentation[i - 1, j])
                               * np.array([-1, 0]))
                        p += p1
                    if j < (compatibility.shape[2] - 1):
                        p2 = compatibility[2, i, j] \
                            * (segmentation[i, j + 1] * np.array([1, 0])
                               + (1 - segmentation[i, j + 1])
                               * np.array([0, 1]))
                        p += p2
                    if j > 0:
                        p3 = compatibility[3, i, j] \
                            * (segmentation[i, j - 1] * np.array([1, 0])
                               + (1 - segmentation[i, j - 1])
                               * np.array([0, 1]))
                        p += p3
                    segmentation[i, j] = np.exp(p[0]) / np.sum(np.exp(p))
        plt.imshow(segmentation)
        plt.colorbar()
        plt.show()
    return segmentation


def segment_EP(compatibility, start, scale=1, steps=5, segment_start=0.499):
    compatibility = compatibility * scale
    segmentation = segment_start * np.ones((compatibility.shape[1],
                                            compatibility.shape[2]))
    segmentation[start[0], start[1]] = 1
    for k in range(steps):
        for i in range(compatibility.shape[1]):
            for j in range(compatibility.shape[2]):
                if not (i == start[0] and j == start[1]):
                    p = np.ones((2, 2, 2, 2, 2))
                    p = p * np.array([1, 2]).reshape((2, 1, 1, 1, 1))
                    if i < (compatibility.shape[1] - 1):
                        p0 = np.exp(0.5 * compatibility[0, i, j]
                                    * np.array([[1, -1], [-1, 1]]))
                        fac_old = p0 * [[segmentation[i, j]],
                                        [1 - segmentation[i, j]]]
                        p0 = p0 / np.sum(fac_old, axis=0, keepdims=True)
                        p0 = p0 * np.array([[segmentation[i + 1, j],
                                             1 - segmentation[i + 1, j]]])
                        p0 = p0.reshape((2, 2, 1, 1, 1))
                        p = p * p0
                    if i > 0:
                        p0 = np.exp(0.5 * compatibility[1, i, j]
                                    * np.array([[1, -1], [-1, 1]]))
                        fac_old = p0 * [[segmentation[i, j]],
                                        [1 - segmentation[i, j]]]
                        p0 /= np.sum(fac_old, axis=0, keepdims=True)
                        p0 *= np.array([[segmentation[i - 1, j],
                                         1 - segmentation[i - 1, j]]])
                        p0 = p0.reshape((2, 1, 2, 1, 1))
                        p = p * p0
                    if j < (compatibility.shape[2] - 1):
                        p0 = np.exp(0.5 * compatibility[2, i, j]
                                    * np.array([[1, -1], [-1, 1]]))
                        fac_old = p0 * [[segmentation[i, j]],
                                        [1 - segmentation[i, j]]]
                        p0 /= np.sum(fac_old, axis=0, keepdims=True)
                        p0 *= np.array([[segmentation[i, j + 1],
                                         1 - segmentation[i, j + 1]]])
                        p0 = p0.reshape((2, 1, 1, 2, 1))
                        p = p * p0
                    if j > 0:
                        p0 = np.exp(0.5 * compatibility[3, i, j]
                                    * np.array([[1, -1], [-1, 1]]))
                        fac_old = p0 * [[segmentation[i, j]],
                                        [1 - segmentation[i, j]]]
                        p0 /= np.sum(fac_old, axis=0, keepdims=True)
                        p0 *= np.array([[segmentation[i, j - 1],
                                         1 - segmentation[i, j - 1]]])
                        p0 = p0.reshape((2, 1, 1, 1, 2))
                        p = p * p0
                    segmentation[i, j] = np.sum(np.sum(np.sum(np.sum(
                        p, axis=4), axis=3), axis=2), axis=1)[0] / np.sum(p)
        for i in range(compatibility.shape[1] - 1, -1, -1):
            for j in range(compatibility.shape[2] - 1, -1, -1):
                if not (i == start[0] and j == start[1]):
                    p = np.ones((2, 2, 2, 2, 2))
                    if i < (compatibility.shape[1]-1):
                        p0 = np.exp(0.5 * compatibility[0, i, j]
                                    * np.array([[1, -1], [-1, 1]]))
                        fac_old = p0 * [[segmentation[i, j]],
                                        [1 - segmentation[i, j]]]
                        p0 /= np.sum(fac_old, axis=0, keepdims=True)
                        p0 *= np.array([[segmentation[i + 1, j],
                                         1 - segmentation[i + 1, j]]])
                        p0 = p0.reshape((2, 2, 1, 1, 1))
                        p = p * p0
                    if i > 0:
                        p0 = np.exp(0.5 * compatibility[1, i, j]
                                    * np.array([[1, -1], [-1, 1]]))
                        fac_old = p0 * [[segmentation[i, j]],
                                        [1 - segmentation[i, j]]]
                        p0 /= np.sum(fac_old, axis=0, keepdims=True)
                        p0 *= np.array([[segmentation[i - 1, j], 1 - segmentation[i - 1, j]]])
                        p0 = p0.reshape((2,1,2,1,1))
                        p = p * p0
                    if j < (compatibility.shape[2]-1):
                        p0 = np.exp(0.5* compatibility[2, i, j] * np.array([[1,-1],[-1,1]]))
                        fac_old = p0 * [[segmentation[i, j]],[1-segmentation[i, j]]]
                        p0 /= np.sum(fac_old, axis=0, keepdims=True)
                        p0 *= np.array([[segmentation[i, j + 1], 1 - segmentation[i, j + 1]]])
                        p0 = p0.reshape((2,1,1,2,1))
                        p = p * p0
                    if j > 0:
                        p0 = np.exp(0.5* compatibility[3, i, j] * np.array([[1,-1],[-1,1]]))
                        fac_old = p0 * [[segmentation[i, j]],[1-segmentation[i, j]]]
                        p0 /= np.sum(fac_old, axis=0, keepdims=True)
                        p0 *= np.array([[segmentation[i, j - 1], 1 - segmentation[i, j - 1]]])
                        p0 = p0.reshape((2,1,1,1,2))
                        p = p * p0
                    segmentation[i, j] = np.sum(np.sum(np.sum(np.sum(p,axis=4),axis=3),axis=2),axis=1)[0]/np.sum(p)
        plt.imshow(segmentation)
        plt.colorbar()
        plt.show()
    return segmentation

def segment_bp(compatibility, start, scale=1, steps = 5, segment_start=0.4):
    compatibility = compatibility * scale
    segmentation = np.array([segment_start * np.ones((compatibility.shape[1], compatibility.shape[2])),
                             (1-segment_start) * np.ones((compatibility.shape[1], compatibility.shape[2]))])
    segmentation[:,start[0], start[1]] = [0.99999, 0.00001]
    factors = np.ones((2,2,2,compatibility.shape[1], compatibility.shape[2]))
    # up down compatibility:
    comp = np.concatenate(((compatibility[0,:-1,:] + compatibility[1,1:,:]) / 2,
                           np.zeros((1, compatibility.shape[2]))))
    factors[0,:,:,:,:] = np.array([[np.exp(comp / 2), np.exp(-comp / 2)],
                                   [np.exp(-comp / 2), np.exp(comp / 2)]])
    # left right compatibility:
    comp = np.concatenate(((compatibility[2,:,:-1] + compatibility[3,:,1:]) / 2,
                           np.zeros((compatibility.shape[1],1))), axis=1)
    factors[1,:,:,:,:] = np.array([[np.exp(comp / 2), np.exp(-comp / 2)],
                                   [np.exp(-comp / 2), np.exp(comp / 2)]])
    # 2 segmentations x 2 factors x 2 values x images size
    messages_s_f = np.ones((2,2,2,compatibility.shape[1], compatibility.shape[2]))
    messages_f_s = np.ones((2,2,2,compatibility.shape[1], compatibility.shape[2]))
    for k in range(steps):
        # update messages segmentation -> factors
        messages_new = 0.5*np.ones((2,2,2,compatibility.shape[1], compatibility.shape[2]))
        messages_new[0,0,:,:,:] = segmentation
        messages_new[0,1,:,:,:] = segmentation
        messages_new[1,0,:,:-1,:] = segmentation[:,1:,:]
        messages_new[1,1,:,:,:-1] = segmentation[:,:,1:]
        messages_new = messages_new / messages_f_s
        messages_new = messages_new / np.sum(messages_new, axis=2, keepdims=True)
        factors = factors * messages_new[0].reshape(
            2,2,1,compatibility.shape[1], compatibility.shape[2])
        factors = factors * messages_new[1].reshape(
            2,1,2,compatibility.shape[1], compatibility.shape[2])
        messages_s_f = messages_new
        # update messages factors -> segmentation
        messages_new = np.ones((2,2,2,compatibility.shape[1], compatibility.shape[2]))
        messages_new[0,:,:,:,:] = np.sum(factors, axis=2)
        messages_new[1,:,:,:,:] = np.sum(factors, axis=1)
        messages_new = messages_new / messages_s_f
        messages_new = messages_new / np.sum(messages_new, axis=2, keepdims=True)
        segmentation = segmentation * messages_new[0,0,:,:,:]
        segmentation = segmentation * messages_new[0,1,:,:,:]
        segmentation[:,1:,:] = segmentation[:,1:,:] * messages_new[1,0,:,:-1,:] 
        segmentation[:,:,1:] = segmentation[:,:,1:] * messages_new[1,1,:,:,:-1]
        messages_f_s = messages_new 
        segmentation = segmentation/np.sum(segmentation, axis=0,keepdims=True)
        factors = factors / np.sum(np.sum(factors, axis = 2, keepdims = True),axis=1,keepdims=True)
        plt.imshow(segmentation[0]/np.sum(segmentation,axis=0))
        plt.colorbar()
        plt.show()

def segment_sampling(compatibility, start, scale=1, steps = 5, segment_start=0.4):
    compatibility = compatibility * scale
    segmentation = (np.random.rand(compatibility.shape[1], compatibility.shape[2])<segment_start).astype(np.float)
    #segmentation = np.zeros((compatibility.shape[1], compatibility.shape[2]))
    segmentation[start[0], start[1]] = 1
    segment_mean = np.zeros((compatibility.shape[1], compatibility.shape[2]))
    for k in range(steps):
        for i in range(compatibility.shape[1]):
            for j in range(compatibility.shape[2]):
                if not (i==start[0] and j==start[1]):
                    p = np.log(np.array([segment_start, 1-segment_start]))
                    if i < (compatibility.shape[1]-1):
                        p0 = compatibility[0, i, j] * (segmentation[i + 1, j] * np.array([1, 0])
                              + (1-segmentation[i + 1, j]) * np.array([-1, 0]))
                        p += p0
                    if i > 0:
                        p1 = compatibility[1, i, j] * (segmentation[i - 1, j] * np.array([1, 0])
                              + (1-segmentation[i - 1, j]) * np.array([-1, 0]))
                        p += p1
                    if j < (compatibility.shape[2]-1):
                        p2 = compatibility[2, i, j] * (segmentation[i, j + 1] * np.array([1, 0])
                              + (1-segmentation[i, j + 1]) * np.array([0, 1]))
                        p += p2
                    if j > 0:
                        p3 = compatibility[3, i, j] * (segmentation[i, j - 1] * np.array([1, 0])
                              + (1-segmentation[i, j - 1]) * np.array([0, 1]))
                        p += p3
                    rand = np.random.rand()
                    segmentation[i, j] = (np.exp(p[0]) / np.sum(np.exp(p))) > rand
        segment_mean += segmentation
        for i in range(compatibility.shape[1]-1,-1,-1):
            for j in range(compatibility.shape[2]-1,-1,-1):
                if not (i==start[0] and j==start[1]):
                    p = np.log(np.array([segment_start, 1-segment_start]))
                    if i < (compatibility.shape[1]-1):
                        p0 = compatibility[0, i, j] * (segmentation[i + 1, j] * np.array([1, 0])
                              + (1-segmentation[i + 1, j]) * np.array([-1, 0]))
                        p += p0
                    if i > 0:
                        p1 = compatibility[1, i, j] * (segmentation[i - 1, j] * np.array([1, 0])
                              + (1-segmentation[i - 1, j]) * np.array([-1, 0]))
                        p += p1
                    if j < (compatibility.shape[2]-1):
                        p2 = compatibility[2, i, j] * (segmentation[i, j + 1] * np.array([1, 0])
                              + (1-segmentation[i, j + 1]) * np.array([0, 1]))
                        p += p2
                    if j > 0:
                        p3 = compatibility[3, i, j] * (segmentation[i, j - 1] * np.array([1, 0])
                              + (1-segmentation[i, j - 1]) * np.array([0, 1]))
                        p += p3
                    rand = np.random.rand()
                    segmentation[i, j] = (np.exp(p[0]) / np.sum(np.exp(p))) > rand
        if not (k % 25):
            plt.imshow(segment_mean)
            plt.colorbar()
            plt.show()
        segment_mean += segmentation
    plt.imshow(segment_mean)
    plt.colorbar()
    plt.show()
    return segment_mean


def segment_sampling_square(compatibility, start, scale=1, steps = 5, segment_start=0.4):
    """ samplessquares of 4 pixels at a time. position of square is indexed
    by the upper left pixel"""
    compatibility = compatibility * scale
    segmentation = (np.random.rand(compatibility.shape[1], compatibility.shape[2])<segment_start).astype(np.float)
    segmentation[start[0], start[1]] = 1
    segment_mean = np.zeros((compatibility.shape[1], compatibility.shape[2]))
    for k in range(steps):
        for i in range(0,compatibility.shape[1]-1,2):
            for j in range(0,compatibility.shape[2]-1,2):
                # indices are upper left, upper right, lower left, lower right
                p = np.zeros((2,2,2,2))
                # single potentials
                p += np.log(np.array([segment_start, 1-segment_start]))
                p += np.log(np.array([[segment_start], [1-segment_start]]))
                p += np.log(np.array([[[segment_start]], [[1-segment_start]]]))
                p += np.log(np.array([[[[segment_start]]], [[[1-segment_start]]]]))
                # start potentials
                if i==start[0] and j==start[1]:
                    p += np.array([[[[10]]], [[[0]]]])
                if i==start[0] and j==start[1]-1:
                    p += np.array([[[[10]], [[0]]]])
                if i==start[0]-1 and j==start[1]:
                    p += np.array([[[[10], [0]]]])
                if i==start[0]-1 and j==start[1]-1:
                    p += np.array([[[[10, 0]]]])
                # pairwise potentials within
                # 0 and 1
                p += np.array([[[[compatibility[2,i,j]]],[[-compatibility[3,i,j+1]]]],
                               [[[-compatibility[3,i,j+1]]],[[compatibility[2,i,j]]]]])/2
                # 0 and 2
                p += np.array([[[[compatibility[0,i,j]],[-compatibility[1,i+1,j]]]],
                               [[[-compatibility[1,i+1,j]],[compatibility[0,i,j]]]]])/2
                # 1 and 3
                p += np.array([[[[compatibility[0,i,j+1],-compatibility[1,i+1,j+1]]],
                                [[-compatibility[1,i+1,j+1],compatibility[0,i,j+1]]]]])/2
                # 2 and 3
                p += np.array([[[[compatibility[2,i+1,j],-compatibility[2,i+1,j]],
                                 [-compatibility[3,i+1,j+1],compatibility[3,i+1,j+1]]]]])/2
                # potentials from above
                if i > 0:
                    p += ((compatibility[0, i-1, j] + compatibility[1, i, j])
                          * np.array([[[[segmentation[i-1,j]]]],[[[1-segmentation[i-1,j]]]]]))/2
                    p += ((compatibility[0, i-1, j+1] + compatibility[1, i, j+1])
                          * np.array([[[[segmentation[i-1,j+1]]],[[1-segmentation[i-1,j+1]]]]]))/2
                # potentials from left
                if j > 0:
                    p += ((compatibility[2, i, j-1] + compatibility[3, i, j])
                          * np.array([[[[segmentation[i,j-1]]]],[[[1-segmentation[i,j-1]]]]]))/2
                    p += ((compatibility[2, i+1, j-1] + compatibility[3, i+1, j])
                          * np.array([[[[segmentation[i+1,j-1]],[1-segmentation[i+1,j-1]]]]]))/2
                # potentials from below
                if i < (compatibility.shape[1]-2):
                    p += ((compatibility[0, i+1, j] + compatibility[1, i+2, j])
                          * np.array([[[[segmentation[i+2,j]],[1-segmentation[i+2,j]]]]]))/2
                    p += ((compatibility[0, i+1, j+1] + compatibility[1, i+2, j+1])
                          * np.array([[[[segmentation[i+2,j+1],1-segmentation[i+2,j+1]]]]]))/2
                # potentials from right
                if j < (compatibility.shape[2]-2):
                    p += ((compatibility[2, i, j+1] + compatibility[3, i, j+2])
                          * np.array([[[[segmentation[i,j+2]]],[[1-segmentation[i,j+2]]]]]))/2
                    p += ((compatibility[2, i+1, j+1] + compatibility[3, i+1, j+2])
                          * np.array([[[[segmentation[i+1,j+2],1-segmentation[i+1,j+2]]]]]))/2
                rand = np.random.rand()
                p_cum = np.cumsum(np.exp(p).flatten())
                p_cum = p_cum / p_cum[-1]
                index = np.where(p_cum > rand)[0][0]
                seg = 1 - np.array(np.unravel_index(index,(2,2,2,2)))
                segmentation[i:(i+2), j:(j+2)] = seg.reshape((2,2))
        segment_mean += segmentation
        if not (k % 5):
            plt.imshow(segment_mean)
            plt.colorbar()
            plt.show()
    plt.imshow(segment_mean)
    plt.colorbar()
    plt.show()
    return segment_mean


def get_simple_compatibiliy(image):
    image = image.astype(np.float)
    im_out = np.zeros([4, image.shape[0], image.shape[1]])
    im_out[0] = conv(image, [0, 1, -1], axis=0, cval=-1,
                     origin=1, mode='constant')
    im_out[1] = conv(image, [1, -1, 0], axis=0, cval=-1,
                     origin=-1, mode='constant')
    im_out[2] = conv(image, [0, 1, -1], axis=1, cval=-1,
                     origin=1, mode='constant')
    im_out[3] = conv(image, [1, -1, 0], axis=1, cval=-1,
                     origin=-1, mode='constant')
    im_out = (abs(im_out) > 0).astype(np.float)
    im_out = 1 - im_out
    return im_out


def graph_spectral_clustering(compatibility):
    from sklearn.cluster import KMeans
    n = compatibility.shape[1] * compatibility.shape[2]
    laplacian = np.zeros((n, n))
    for i in range(compatibility.shape[1]):
        for j in range(compatibility.shape[2]):
            if i < (compatibility.shape[1]-1):
                laplacian[compatibility.shape[1] * i + j,
                          compatibility.shape[1] * (i + 1) + j] += \
                    compatibility[0, i, j]
            if i > 0:
                laplacian[compatibility.shape[1] * i + j,
                          compatibility.shape[1] * (i - 1) + j] += \
                    compatibility[1, i, j]
            if j < (compatibility.shape[1] - 1):
                laplacian[compatibility.shape[1] * i + j,
                          compatibility.shape[1] * i + j + 1] += \
                    compatibility[2, i, j]
            if j > 0:
                laplacian[compatibility.shape[1] * i + j,
                          compatibility.shape[1] * i + j - 1] += \
                    compatibility[3, i, j]
    laplacian = (laplacian + laplacian.T) / 2
    diag = -np.sum(laplacian, axis=1)
    np.fill_diagonal(laplacian, diag)
    laplacian = laplacian / (diag + eps)
    w, v = scipy.linalg.eig(laplacian, b=np.diag(-diag + eps))
    thresh = 10 * eps
    v_used = v[:, np.abs(w) < thresh]
    kmeans = KMeans(n_clusters=v_used.shape[1])
    kmeans.fit(v_used)
    plt.figure()
    plt.plot(np.sort(np.abs(w)))
    plt.show()
    return kmeans.labels_.reshape((compatibility.shape[1],
                                   compatibility.shape[2]))

def segment_flood(compatibility, start, scale=1, steps = 5, segment_start=0.4):
    compatibility = compatibility * scale
    segmentation = np.zeros((compatibility.shape[1], compatibility.shape[2]))
    divisor = np.sum(compatibility, axis=0)
    divisor[divisor == 0] = 1
    for k in range(steps):
        segmentation[start[0], start[1]] += 1
        flow = compatibility / divisor * segmentation
        flow[0] = np.concatenate((np.zeros((1,compatibility.shape[2])),
                                  flow[0,:-1,:]),
                                 axis=0)
        flow[1] = np.concatenate((flow[1,1:,:],
                                  np.zeros((1,compatibility.shape[2]))),
                                  axis=0)
        flow[2] = np.concatenate((np.zeros((compatibility.shape[1],1)),
                                  flow[2,:,:-1]), 
                                 axis=1)
        flow[3] = np.concatenate((flow[3,:,1:],
                                  np.zeros((compatibility.shape[1],1))), 
                                 axis=1)
        segmentation = np.sum(flow,axis=0)
        if not (k % 5):
            plt.imshow(segmentation/(k+1))
            plt.colorbar()
            plt.show()
    plt.imshow(segmentation/(k+1))
    plt.colorbar()
    plt.show()
    return segmentation


import PIL.Image
im = PIL.Image.open('/Users/heiko/deadrects/validation_30/image0000025.png')
image = np.array(im)[:, :, 1]
compatibility = get_simple_compatibiliy(image)
compatibility = compatibility + 0.01 * np.random.rand(compatibility.shape[0],
                                                      compatibility.shape[1],
                                                      compatibility.shape[2])
seg = segment_flood(compatibility, [5,20], steps=100)

if False:
    #seg_final = segment_sampling(compatibility, [4,2], scale=2, steps = 500, segment_start=.45)
    #seg_final = segment_sampling_square(compatibility, [4,2], scale=2, steps = 20, segment_start=.48)
    t1 = time.time()
    
    cluster_im = graph_spectral_clustering(compatibility)
    
    t2 = time.time()
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(cluster_im)
    
    t3 = time.time()
    print(t1-t0)
    print(t2-t1)
    print(t3-t2)
    
if False:
    for i in range(compatibility.shape[1]):
        for j in range(compatibility.shape[2]):
            p = np.array([0.4,0.6])
            if i < (compatibility.shape[1]-1):
                p0 = (compatibility[0, i + 1, j] * segmentation[i + 1, j] * np.array([1, 0])
                  + (1-compatibility[0, i + 1, j]) * 0.5
                  + (compatibility[0, i + 1, j] * (1-segmentation[i + 1, j]) * np.array([0, 1])))
                p *= p0
            if j < (compatibility.shape[2]-1):
                p1 = (compatibility[1, i, j + 1] * segmentation[i, j + 1] * np.array([1, 0.0001])
                  + (1-compatibility[1, i, j + 1]) * 0.5
                  + (compatibility[1, i, j + 1] * (1-segmentation[i, j + 1]) * np.array([0.0001, 1])))
                p *= p1
            if i > 0:
                p2 = (compatibility[1, i - 1, j] * segmentation[i - 1, j] * np.array([1, 0.0001])
                  + (1-compatibility[1, i - 1, j]) * 0.5
                  + (compatibility[1, i - 1, j] * (1-segmentation[i - 1, j]) * np.array([0.0001, 1])))
                p *= p2
            if j > 0:
                p3 = (compatibility[1, i, j - 1] * segmentation[i, j - 1] * np.array([1, 0.0001])
                  + (1-compatibility[1, i, j - 1]) * 0.5
                  + (compatibility[1, i, j - 1] * (1-segmentation[i, j - 1]) * np.array([0.0001, 1])))
                p *= p3
            segmentation[i,j] = p[0] / np.sum(p)
    segmentation[start[0], start[1]] = 1
    for i in range(compatibility.shape[1]-1,-1,-1):
        for j in range(compatibility.shape[2]-1,-1,-1):
            p = np.array([0.4,0.6])
            if i < (compatibility.shape[1]-1):
                p0 = (compatibility[0, i + 1, j] * segmentation[i + 1, j] * np.array([1, 0.0001])
                  + (1-compatibility[0, i + 1, j]) * 0.5
                  + (compatibility[0, i + 1, j] * (1-segmentation[i + 1, j]) * np.array([0.0001, 1])))
                p *= p0
            if j < (compatibility.shape[2]-1):
                p1 = (compatibility[1, i, j + 1] * segmentation[i, j + 1] * np.array([1, 0.0001])
                  + (1-compatibility[1, i, j + 1]) * 0.5
                  + (compatibility[1, i, j + 1] * (1-segmentation[i, j + 1]) * np.array([0.0001, 1])))
                p *= p1
            if i > 0:
                p2 = (compatibility[1, i - 1, j] * segmentation[i - 1, j] * np.array([1, 0.0001])
                  + (1-compatibility[1, i - 1, j]) * 0.5
                  + (compatibility[1, i - 1, j] * (1-segmentation[i - 1, j]) * np.array([0.0001, 1])))
                p *= p2
            if j > 0:
                p3 = (compatibility[1, i, j - 1] * segmentation[i, j - 1] * np.array([1, 0.0001])
                  + (1-compatibility[1, i, j - 1]) * 0.5
                  + (compatibility[1, i, j - 1] * (1-segmentation[i, j - 1]) * np.array([0.0001, 1])))
                p *= p3
            segmentation[i,j] = p[0] / np.sum(p)

