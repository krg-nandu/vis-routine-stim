#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:33:07 2019

@author: heiko
"""
import numpy as np
import pandas as pd
from skimage import io 
import os
import tqdm

import DeadLeaf as dl


nCols = 9
imSize=np.array((5,5))
distances=None
sizes=np.arange(1,6)

cols = (255*np.linspace(0,1,nCols)).astype(np.uint8)

# Analysis of the image

def main(idx=0,root_dir = '/Users/heiko/tinytinydeadrects/training',exponent = 3):
    outdir = root_dir + 'Bayes'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    solutions_df = pd.read_csv(os.path.join(root_dir,'solution.csv'),index_col=0)
    img_name = os.path.join(root_dir,solutions_df['im_name'].iloc[idx])
    test_image = io.imread(img_name).astype(np.float32)
    test_image = np.array(test_image.transpose([2,0,1]))
    prob = dl.get_default_prob(exponent,sizes=sizes)
    g = dl.graph(test_image[2],sizes=sizes,colors=cols,prob = prob)
    
    points = (test_image[0]==255) & (test_image[1]==0)
    points = np.where(points)
    #g.get_exact_prob(points)
    
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
    data = np.stack((estimate,logPPos,logPVis,logPCorrection))
    file_out = outdir+os.path.sep+ 'samples%07d.npy' % idx
    if os.path.isfile(file_out):
        data_load = np.load(file_out)
        data = np.concatenate((data_load,data), axis = 1)
    np.save(outdir+os.path.sep+ 'samples%07d.npy' % idx,data)
    
    lik = np.array(logPPos)-np.array(logPVis)+np.array(logPCorrection)
    lik = np.exp(lik-np.max(lik))
    lik = lik/np.sum(lik)
    
    p_same = np.sum(lik*np.array(estimate))
    return p_same
    
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--index", help="which image to analyze", type = int,default=0)
    parser.add_argument("-p","--path", help="directory containing the images", type = str ,default='/Users/heiko/tinytinydeadrects/training')
    args=parser.parse_args()
    main(idx = args.index, root_dir = args.path)
    