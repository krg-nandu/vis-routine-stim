#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:28:09 2019
Gibbs sampling with fixed noise
@author: heiko
"""

import numpy as np
import scipy.signal

x = np.random.randn(2)

mu = np.array([1, 0])
sigma = np.array([1, 5])
sigma12 = 1
cycles = 10

xs = []
for i_samp in range(5000):
    xs.append(x.copy())
    x = mu+np.random.randn(2)/np.sqrt(sigma)
    for iConverge in range(cycles):
        for i in range(len(x)):
            mu_new = (sigma[i]*mu[i]+sigma12*x[-i-1])/(sigma[i]+sigma12)
            sigma_new = sigma[i]+sigma12
            x[i] = mu_new + np.random.randn(1)/np.sqrt(sigma_new)

xs = np.array(xs)
print(np.cov(xs[:, 0], xs[:, 1]))


autocorr = [np.correlate(xs[:, 0], xs[:, 0])]

for iLag in range(1, 20):
    autocorr.append(np.correlate(xs[iLag:, 0], xs[:(-iLag), 0]))

precision = np.array([[sigma[0] + sigma12, -sigma12],
                      [-sigma12, sigma[1] + sigma12]])
print(np.linalg.inv(precision))

cholesky = np.linalg.cholesky(np.linalg.inv(precision))

y = np.matmul(cholesky, np.random.randn(2, 5000)).T


def get_gibbs_sample(im_size, prec, prec_0, N):
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    prec_samp = scipy.signal.convolve2d(np.ones((im_size[0], im_size[1])),
                                        kernel, 'same') * prec + prec_0
    image = np.random.randn(im_size[0], im_size[1]) / prec_samp
    for i in range(N):
        selection = (np.random.rand(im_size[0], im_size[1]) > .5)
        mean = (scipy.signal.convolve2d(image, kernel, 'same') *
                prec / prec_samp)
        image_new = mean + (np.random.randn(im_size[0], im_size[1]) /
                            np.sqrt(prec_samp))
        image[selection] = image_new[selection]
        mean = (scipy.signal.convolve2d(image, kernel, 'same') *
                prec / prec_samp)
        image_new = mean + (np.random.randn(im_size[0], im_size[1]) /
                            np.sqrt(prec_samp))
        image[~selection] = image_new[~selection]
    return image


def get_gibbs_sample_proper(im_size, prec, prec_0, N):
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    prec_samp = prec * scipy.signal.convolve2d(np.ones((im_size[0], im_size[1])),
                                               kernel, 'same') + prec_0
    image = np.random.randn(im_size[0], im_size[1]) / prec_samp
    for k in range(N):
        for i in range(im_size[0]):
            for j in range(im_size[1]):
                mu = 0
                if i > 0:
                    mu = mu + image[i-1, j]
                if j > 0:
                    mu = mu + image[i, j-1]
                if i < (im_size[0] - 1):
                    mu = mu + image[i+1, j]
                if j < (im_size[1] - 1):
                    mu = mu + image[i, j+1]
                image[i, j] = ((prec * mu)/prec_samp[i, j] +
                               np.random.randn() / np.sqrt(prec_samp[i, j]))
    return image
