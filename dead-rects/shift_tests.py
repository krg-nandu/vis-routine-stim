#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:58:17 2019

@author: heiko
"""


import sys, getopt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tqdm


def get_shifted_values(feat,neighbors):
    output = torch.zeros(feat.shape).repeat([neighbors.shape[0],1,1,1,1])
    for iNeigh in range(neighbors.shape[0]):
        if neighbors[iNeigh,0]>= 0 and neighbors[iNeigh,1] >= 0:
            output[iNeigh,:,:,:(feat.shape[2]-int(neighbors[iNeigh,0])),:(feat.shape[3]-int(neighbors[iNeigh,1]))] = feat[:,:,int(neighbors[iNeigh,0]):,int(neighbors[iNeigh,1]):]
        elif neighbors[iNeigh,0]>= 0 and neighbors[iNeigh,1] < 0:
            output[iNeigh,:,:,:(feat.shape[2]-int(neighbors[iNeigh,0])),int(-neighbors[iNeigh,1]):] = feat[:,:,int(neighbors[iNeigh,0]):,:(feat.shape[3]-int(-neighbors[iNeigh,1]))]
        elif neighbors[iNeigh,0]< 0 and neighbors[iNeigh,1] >= 0:
            output[iNeigh,:,:,int(-neighbors[iNeigh,0]):,:(feat.shape[3]-int(neighbors[iNeigh,1]))] = feat[:,:,:(feat.shape[2]-int(-neighbors[iNeigh,0])),int(neighbors[iNeigh,1]):]
        elif neighbors[iNeigh,0]< 0 and neighbors[iNeigh,1] < 0:
            output[iNeigh,:,:,int(-neighbors[iNeigh,0]):,int(-neighbors[iNeigh,1]):] = feat[:,:,:(feat.shape[2]-int(-neighbors[iNeigh,0])),:(feat.shape[3]-int(-neighbors[iNeigh,1]))]
    return output    

def get_shifted_values_conv(feat,neighbors):
    k = np.max(np.abs(neighbors))
    w = torch.zeros(len(neighbors)*feat.shape[1],feat.shape[1],2*k+1,2*k+1,requires_grad = False)
    for iNeigh in range(neighbors.shape[0]):
        w[(iNeigh*feat.shape[1]):((iNeigh+1)*feat.shape[1]),:,neighbors[iNeigh,0]+k,neighbors[iNeigh,1]+k]=torch.eye(feat.shape[1])
    output = torch.conv2d(feat,w,padding=(k,k))
    output = output.permute(1,0,2,3).reshape(neighbors.shape[0],feat.shape[1],feat.shape[0],feat.shape[2],feat.shape[3]).permute(0,2,1,3,4)
    return output  


def get_shifted_values_conv2(feat,neighbors):
    k = np.max(np.abs(neighbors))
    w = torch.zeros(len(neighbors),1,2*k+1,2*k+1,requires_grad = False)
    for iNeigh in range(neighbors.shape[0]):
        w[iNeigh,:,neighbors[iNeigh,0]+k,neighbors[iNeigh,1]+k]=1
    output = torch.conv2d(feat.reshape(feat.shape[1]*feat.shape[0],1,feat.shape[2],feat.shape[3]),w,padding=(k,k))
    output = output.permute(1,0,2,3).reshape(neighbors.shape[0],feat.shape[0],feat.shape[1],feat.shape[2],feat.shape[3])
    return output  

def get_shifted_values3(feat,neighbors):
    out = []
    for iNeigh in range(neighbors.shape[0]):
        output = torch.zeros(feat.shape)
        if neighbors[iNeigh,0]>= 0 and neighbors[iNeigh,1] >= 0:
            output[:,:,:(feat.shape[2]-int(neighbors[iNeigh,0])),:(feat.shape[3]-int(neighbors[iNeigh,1]))] = feat[:,:,int(neighbors[iNeigh,0]):,int(neighbors[iNeigh,1]):]
        elif neighbors[iNeigh,0]>= 0 and neighbors[iNeigh,1] < 0:
            output[:,:,:(feat.shape[2]-int(neighbors[iNeigh,0])),int(-neighbors[iNeigh,1]):] = feat[:,:,int(neighbors[iNeigh,0]):,:(feat.shape[3]-int(-neighbors[iNeigh,1]))]
        elif neighbors[iNeigh,0]< 0 and neighbors[iNeigh,1] >= 0:
            output[:,:,int(-neighbors[iNeigh,0]):,:(feat.shape[3]-int(neighbors[iNeigh,1]))] = feat[:,:,:(feat.shape[2]-int(-neighbors[iNeigh,0])),int(neighbors[iNeigh,1]):]
        elif neighbors[iNeigh,0]< 0 and neighbors[iNeigh,1] < 0:
            output[:,:,int(-neighbors[iNeigh,0]):,int(-neighbors[iNeigh,1]):] = feat[:,:,:(feat.shape[2]-int(-neighbors[iNeigh,0])),:(feat.shape[3]-int(-neighbors[iNeigh,1]))]
        out.append(output)
    output = torch.cat(out,0).reshape(neighbors.shape[0],feat.shape[0],feat.shape[1],feat.shape[2],feat.shape[3])
    return output  

def test_run(version = 0):
    feat = torch.rand(6,10,300,300,requires_grad = True)
    neighbors=np.array([[0,1],[0,-1],[1,0],[-1,0]])
    if version == 0:
        a = get_shifted_values(feat,neighbors)
    elif version == 2:
        a = get_shifted_values_conv2(feat,neighbors)
    elif version == 3:
        a = get_shifted_values3(feat,neighbors)
    else:
        a = get_shifted_values_conv(feat,neighbors)
    b = torch.sum(a)
    b.backward()


test_run(version=4)