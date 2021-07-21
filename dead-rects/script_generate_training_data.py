#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:22:31 2019

@author: heiko
"""

import DeadLeaf as dl
import numpy as np

## first: exactly as in experiment
dl.save_training_data('training',100000)
dl.save_training_data('validation',10000)

## separate by exponent
for i in range(1,6):
    dl.save_training_data('training%d' % i,100000,exponents=np.array(i))
    dl.save_training_data('validation%d' % i,10000,exponents=np.array(i))
    
## full range of distances
distances = np.arange(2,250)
distancesd = np.arange(2,250)
dl.save_training_data('trainingAll',100000,distances=distances,distancesd=distancesd)
dl.save_training_data('validationAll',10000,distances=distances,distancesd=distancesd)
