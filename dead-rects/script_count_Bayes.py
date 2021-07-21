#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:10:53 2019

@author: heiko
"""

import numpy as np
import os
import pandas as pd

root_dir = '/Users/heiko/tinytinydeadrects/validation'

files = os.listdir(root_dir+'Bayes')
files.sort()

ps = []
n_eff = []

for i, file in enumerate(files):
    samples = np.load(root_dir+'Bayes'+os.path.sep+file)
    estimate = samples[0]
    logPPos = samples[1]
    logPVis = samples[2]
    logPCorrection = samples[3]
    lik = np.array(logPPos)-np.array(logPVis)+np.array(logPCorrection)
    lik = np.exp(lik-np.max(lik))
    lik = lik/np.sum(lik)
    p_same = np.sum(lik*np.array(estimate))
    ps.append(p_same)
    n_eff.append(1/np.sum(lik**2))
ps = np.array(ps)
n_eff = np.array(n_eff)

solution_df = pd.read_csv(root_dir+os.path.sep+'solution.csv')
solution = np.array(solution_df['solution'])[:1000]

