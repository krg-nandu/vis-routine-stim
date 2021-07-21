#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:53:57 2019

@author: heiko
"""


import DeadLeaf as dl
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import os.path

def main(n_images = 1000000, im_size=3, n_val = 10000, n_test = 10000, n_col=9):
    if im_size == 3:
        sizes = np.array([1,3,5])
        exponents = np.array([3])
    elif im_size == 5:
        sizes = np.array([1,3,5])
        exponents = np.array([3])
    else:
        sizes = np.arange(5,im_size+1,5)
        exponents = np.array([3])
    im_size = np.array((im_size,im_size))

    #p_dist = np.load('p_dist_5.npy')
    p_dist, p_same_sum, p_same = optimize_distance_distribution2(im_size, sizes, exponents, n_col=n_col)
    plt.figure()
    plt.imshow(p_dist, vmin=0, vmax=np.max(p_dist))
    plt.title(p_same_sum)
    plt.colorbar()
    # plt.show()

    if n_col==9:
        np.save('/Users/heiko/deadrects/p_dist_%d' % im_size[0], p_dist)
        np.save('/Users/heiko/deadrects/p_same_%d' % im_size[0], p_same)
        np.save('/Users/heiko/deadrects/p_same_sum_%d' % im_size[0], p_same_sum)

        if not os.path.isfile('/Users/heiko/deadrects/training_%d/solution.csv' % im_size[0]):
            dl.save_training_data('/Users/heiko/deadrects/training_%d' % im_size[0],
                                  n_images, im_size=im_size, sizes=sizes, exponents=exponents,
                                  dist_probabilities=p_dist, same_color=2)
        if not os.path.isfile('/Users/heiko/deadrects/validation_%d/solution.csv' % im_size[0]):
            dl.save_training_data('/Users/heiko/deadrects/validation_%d' % im_size[0],
                                  n_val, im_size=im_size, sizes=sizes,
                                  exponents=exponents, dist_probabilities=p_dist,
                                  same_color=2)
        if not os.path.isfile('/Users/heiko/deadrects/test_%d/solution.csv' % im_size[0]):
            dl.save_training_data('/Users/heiko/deadrects/test_%d' % im_size[0],
                                  n_test, im_size=im_size, sizes=sizes,
                                  exponents=exponents, dist_probabilities=p_dist,
                                  same_color=2)
    else:
        np.save('/Users/heiko/deadrects/p_dist_%d_C%d' % (im_size[0], n_col),p_dist)
        np.save('/Users/heiko/deadrects/p_same_%d_C%d' % (im_size[0], n_col),p_same)
        np.save('/Users/heiko/deadrects/p_same_sum_%d_C%d' % (im_size[0], n_col),p_same_sum)

        if not os.path.isfile('/Users/heiko/deadrects/training_%d_C%d/solution.csv' % (im_size[0], n_col)):
            dl.save_training_data('/Users/heiko/deadrects/training_%d_C%d' % (im_size[0], n_col),
                                  n_images, im_size=im_size, sizes=sizes, exponents=exponents,
                                  dist_probabilities=p_dist, same_color=2)
        if not os.path.isfile('/Users/heiko/deadrects/validation_%d_C%d/solution.csv' % (im_size[0], n_col)):
            dl.save_training_data('/Users/heiko/deadrects/validation_%d_C%d' % (im_size[0], n_col),
                                  n_val, im_size=im_size, sizes=sizes,
                                  exponents=exponents, dist_probabilities=p_dist,
                                  same_color=2)
        if not os.path.isfile('/Users/heiko/deadrects/test_%d_C%d/solution.csv' % (im_size[0], n_col)):
            dl.save_training_data('/Users/heiko/deadrects/test_%d_C%d' % (im_size[0], n_col),
                                  n_test, im_size=im_size, sizes=sizes,
                                  exponents=exponents, dist_probabilities=p_dist,
                                  same_color=2)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    From: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def calc_prob_one(sizes = [5,10,15],grid=None,prob=None,dx = 1,dy = 1):
    sizes = np.array(sizes)
    if grid is not None:
        sizes = sizes/grid
        dx = dx/grid
        dy = dy/grid
    if prob is None:
        prob = np.ones(len(sizes))
    if len(sizes.shape) == 1:
        sizes = cartesian([sizes,sizes])
        prob = np.outer(prob,prob).flatten()
    p1 = sum(prob*np.array([max(0,(sizes[k,0]-dx))*max(0,(sizes[k,1]-dy)) for k in range(len(sizes))]))
    p2 = sum(prob*np.array([2*sizes[k,0]*sizes[k,1]-2*max(0,(sizes[k,0]-dx))*max(0,(sizes[k,1]-dy)) for k in range(len(sizes))]))
    p = p1/(p1+p2)
    return p
    #return (p,p1,p2)

def calc_prob_one_grid(sizes = [5,10,15],grid=None,prob = None,dx = 1,dy = 1):
    ps = np.zeros((len(dx),len(dy)))
    kx = 0
    for idx in tqdm.tqdm(dx):
        ky=0
        for idy in dy:
            ps[kx,ky] = calc_prob_one(sizes = sizes, grid = grid, prob = prob, dx = idx,dy = idy)
            ky += 1
        kx += 1
    return ps

def calc_distance_distribution(ps,weights):
    ps = ps.flatten()
    p_tile = ps.repeat((len(ps)),1)
    p_tile = p_tile * (1-torch.eye(len(ps)))
    p_tile = p_tile/torch.sum(p_tile,dim=1).view(-1,1)
    p_tile = p_tile * ps.view(-1,1)
    p_same_sum = torch.sum(p_tile * weights)
    return p_same_sum

def optimize_distance_distribution(im_size,sizes,exponents):
    print('started optimizing the distance distribution\n')
    ps = torch.ones((im_size[0],im_size[1]))/np.prod(im_size)
    ps = ps.flatten()[1:]
    ps = torch.log(ps)
    ps.requires_grad=True
    ps.data = ps - torch.logsumexp(ps,dim=0)

    p_best = torch.ones((im_size[0],im_size[1]))/np.prod(im_size)
    p_best[0] = p_best[0]/2
    p_best[:,0] = p_best[:,0]/2
    p_best = p_best.flatten()[1:]
    entropy = torch.sum(p_best * torch.log(p_best))

    print('calculating p_same\n')
    p_same = np.zeros(im_size)
    for iExp in exponents:
        prob = sizes ** (-iExp/2)
        prob = prob/np.sum(prob)
        p_same += calc_prob_one_grid(sizes = sizes, prob = prob, grid = None, dx = np.arange(im_size[0]), dy = np.arange(im_size[1]))

    p_same = torch.Tensor(p_same/len(exponents))

    plt.figure()
    plt.imshow(p_same)
    plt.colorbar()

    p_same_vec = p_same.flatten()[1:]

    optimizer = torch.optim.SGD([ps], lr = 0.1, momentum = 0)

    print('starting main optimizatiton\n')
    if im_size[0] <=100:
        N = 15000
    else:
        N = 150000
    losses = []
    for iUpdate in tqdm.trange(N):
        p_same_sum = torch.sum(torch.exp(ps-torch.logsumexp(ps,0))*p_same_vec)

        loss = 10*(torch.abs(p_same_sum-0.5)) - torch.sum(p_best * (ps-torch.logsumexp(ps,0))) + entropy

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        ps.data = ps - torch.logsumexp(ps,dim=0)

        optimizer.param_groups[0]['lr'] =  5/(iUpdate+5)
        losses.append(loss.item())

    ps_im = np.concatenate(([0], ps.exp().detach().numpy())).reshape(im_size)
    return ps_im, p_same_sum.detach().numpy(), p_same.numpy()

def optimize_distance_distribution2(im_size,sizes,exponents,n_col=9):
    print('started optimizing the distance distribution\n')
    lamb1 = torch.Tensor([-1])
    lamb1.requires_grad = True

    print('calculating p_same\n')
    p_same = np.zeros(im_size)
    for iExp in exponents:
        prob = sizes ** (-iExp/2)
        prob = prob/np.sum(prob)
        p_same += calc_prob_one_grid(sizes = sizes, prob = prob, grid = None, dx = np.arange(im_size[0]), dy = np.arange(im_size[1]))

    p_same = p_same/len(exponents)

    p_same = (n_col*p_same)/((n_col-1)*p_same+1)
    #p_same = torch.Tensor(p_same/len(exponents))
    p_same_full = np.concatenate(
            [np.flip(np.concatenate([np.flip(p_same[1:,1:],0),p_same[:,1:]]),1),
             np.concatenate([np.flip(p_same[1:],0),p_same])],1)

    plt.figure()
    plt.imshow(p_same)
    plt.colorbar()

    p_same_vec = p_same_full.flatten()
    id1 = int(2*(im_size[1]-1)*im_size[0])
    p_same_vec = np.concatenate([p_same_vec[:id1],
                                p_same_vec[(id1+1):]])

    p_same_vec = torch.Tensor(p_same_vec)

    lamb1 = lamb1-1
    ps = -lamb1*p_same_vec
    ps_norm = ps - torch.logsumexp(ps,dim=0)
    p_same_sum = torch.sum(torch.exp(ps_norm)*p_same_vec)

    lamb1 = 0
    while p_same_sum>0.5:
        lamb1 = lamb1+1
        ps = -lamb1*p_same_vec
        ps_norm = ps - torch.logsumexp(ps,dim=0)
        p_same_sum = torch.sum(torch.exp(ps_norm)*p_same_vec)
    while p_same_sum<0.5:
        lamb1 = lamb1-1
        ps = -lamb1*p_same_vec
        ps_norm = ps - torch.logsumexp(ps,dim=0)
        p_same_sum = torch.sum(torch.exp(ps_norm)*p_same_vec)
    lamb2 = lamb1+1
    N = 50
    for iUpdate in range(N):
        lamb_new = (lamb1+lamb2)/2
        ps = -lamb_new*p_same_vec
        ps_norm = ps - torch.logsumexp(ps,dim=0)
        p_same_sum = torch.sum(torch.exp(ps_norm)*p_same_vec)
        if p_same_sum > 0.5:
            lamb1 = lamb_new
        else:
            lamb2 = lamb_new
#
#    optimizer = torch.optim.SGD([lamb1], lr = 0.1, momentum = 0)
#
#    print('starting main optimizatiton\n')
#    if im_size[0] <=100:
#        N = 15000
#    else:
#        N = 150000
#    losses = []
#    for iUpdate in tqdm.trange(N):
#        ps = -lamb1*p_same_vec
#        ps_norm = ps - torch.logsumexp(ps,dim=0)
#        p_same_sum = torch.sum(torch.exp(ps_norm)*p_same_vec)
#
#        loss = torch.abs(p_same_sum-0.5)
#
#        loss.backward()
#
#        optimizer.step()
#        optimizer.zero_grad()
#
#        #optimizer.param_groups[0]['lr'] =  5/(iUpdate+5)
#        losses.append(loss.item())

    lamb_new = (lamb1+lamb2)/2
    ps = -lamb_new*p_same_vec
    ps_norm = ps - torch.logsumexp(ps,dim=0)
    ps = torch.exp(ps_norm).detach().numpy()
    ps_im = np.concatenate([ps[:id1],[0],ps[(id1):]])
    ps_im = ps_im.reshape(2*np.array(im_size)-1)
    ps_im = 4*ps_im[(im_size[0]-1):,(im_size[1]-1):]
    ps_im[0] = ps_im[0]/2
    ps_im[:,0] = ps_im[:,0]/2

    #ps_im = np.concatenate(([0], ps.exp().detach().numpy())).reshape(im_size)
    return ps_im, p_same_sum.detach().numpy(), p_same

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--n_images", help="numer of training images", type = int ,default=1000000)
    parser.add_argument("-v","--n_val", help="numer of validation images", type = int ,default=10000)
    parser.add_argument("-i","--im_size",type=int,help="image size",default=5)
    parser.add_argument("-c","--n_col",type=int,help="number of colors",default=9)
    args=parser.parse_args()
    main(n_images=args.n_images, im_size=args.im_size, n_val=args.n_val, n_col=args.n_col)
