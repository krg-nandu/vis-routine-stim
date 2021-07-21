# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:03:12 2018

@author: heiko
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter
import tqdm
import pandas as pd
from skimage import io
import cv2


default_sizes = 5 * np.arange(1, 80, dtype='float')
default_colors_1 = np.linspace(0, 1, 9)
default_colors_255 = np.array([0,  31,  63,  95, 127, 159, 191, 223, 255], dtype=np.uint8)


def get_default_prob(exponent, sizes=default_sizes):
    return (sizes / np.min(sizes)) ** -(exponent / 2)


def gen_rect_leaf(im_size=[255, 255], sizes=[5, 10, 15],
                  colors=[0, 0.5, 1], grid=1, noise=0,
                  noiseType='norm', prob=None, fixedC=0, fixedIdx=[],
                  border=False):
    if prob is None:
        prob = np.ones(len(sizes))
    assert (im_size[0] % grid) == 0, 'Image Size not compatible with grid'
    assert (im_size[1] % grid) == 0, 'Image Size not compatible with grid'
    fixedIdx = np.array(fixedIdx)
    sizes = np.array(sizes)
    prob = np.array(prob)
    assert np.all(np.array(sizes) % grid == 0), 'Patch sizes not compatible with grid'
    assert noise >= 0, 'noise is the standard deviation and thus should be >=0'
    assert np.all(prob > 0), 'probabilities for shapes must be >0'
    assert prob.size == sizes.size, 'probabilities and sizes should have equal length'

    # correction for the different size of the possible area
    if len(np.array(sizes).shape) == 1:
        probx = prob * (sizes+im_size[0]-1)/(np.max(sizes)+im_size[0]-1)
        proby = prob * (sizes+im_size[1]-1)/(np.max(sizes)+im_size[1]-1)
        probx = probx/np.sum(probx)
        proby = proby/np.sum(proby)
        probcx = probx.cumsum()
        probcy = proby.cumsum()
    else:
        prob = prob * (sizes[:, 0] + im_size[0] - 1) / (np.max(sizes[:, 0]) + im_size[0] - 1)
        prob = prob * (sizes[:, 1] + im_size[1] - 1) / (np.max(sizes[:, 1]) + im_size[1] - 1)
        prob = prob/np.sum(prob)
        probc = prob.cumsum()
    image = np.nan * np.zeros(im_size, dtype='float')
    rectList = list()
    while np.any(np.isnan(image)):
        if len(np.array(sizes).shape) == 1:
            idx_sizex = np.searchsorted(probcx, np.random.rand())
            idx_sizey = np.searchsorted(probcy, np.random.rand())
            sizx = sizes[idx_sizex]
            sizy = sizes[idx_sizey]
        elif len(np.array(sizes).shape) == 2:
            idx_size = np.searchsorted(probc, np.random.rand())
            sizx = sizes[idx_size][0]
            sizy = sizes[idx_size][1]
        idx_color = np.random.randint(len(colors))
        c = colors[idx_color]
        sizx = sizx/grid
        sizy = sizy/grid
        idx_x = np.random.randint(1 - sizx, im_size[0] / grid)
        idx_y = np.random.randint(1 - sizy, im_size[1] / grid)
        rectList.append([grid * idx_x, grid * idx_y, grid * sizx, grid * sizy, idx_color])
        image[int(grid * max(idx_x, 0)):int(grid * max(0, idx_x + sizx)),
              int(grid * max(idx_y, 0)):int(grid * max(0, idx_y + sizy))] = c
    rectList = np.array(rectList, dtype=np.int16)
    oneObject = False
    # if we want to fix some point in the image to a color
    # find last rectangle put in these places and redraw all rectangles up to this point
    idxStart = -1
    while len(fixedIdx) > 0:
        idxStart = idxStart+1
        R = rectList[idxStart]
        delete = []
        for i in range(len(fixedIdx)):
            # print((R[0]<=fixedIdx[i][0]),((R[0]+R[2])>fixedIdx[i][0]),(R[1]<=fixedIdx[i][1]),((R[1]+R[3])>fixedIdx[i][1]))
            if (R[0] <= fixedIdx[i][0]) and ((R[0] + R[2]) > fixedIdx[i][0]) \
                and (R[1] <= fixedIdx[i][1]) and ((R[1] + R[3]) > fixedIdx[i][1]):
                rectList[idxStart, -1] = fixedC
                delete.append(i)
                # print('found one')
        if len(delete) > 1:
            oneObject = True
        fixedIdx = np.delete(fixedIdx, delete, axis=0)
    for i in range(len(rectList)):
        image[int(max(rectList[len(rectList) - i - 1, 0], 0)):int(
            max(0, rectList[len(rectList) - i - 1, 0] + rectList[len(rectList) - i - 1, 2])),
              int(max(rectList[len(rectList) - i - 1, 1], 0)):int(
            max(0, rectList[len(rectList) - i - 1, 1] + rectList[len(rectList) - i - 1, 3]))] \
            = colors[rectList[len(rectList) - i - 1, -1]]
        if border:
            idx_x = rectList[len(rectList) - i - 1, 0]
            idx_y = rectList[len(rectList) - i - 1, 1]
            sizx = rectList[len(rectList) - i - 1, 2]
            sizy = rectList[len(rectList) - i - 1, 3]
            if idx_x >= 0:
                image[int(idx_x), int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = 5
            if (idx_x + sizx) <= im_size[0]:
                image[int((idx_x + sizx)-1), int(max(idx_y, 0)):int(idx_y + sizy)] = 5
            if idx_y >= 0:
                image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y)] = 5
            if (idx_y+sizy) <= im_size[1]:
                image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y + sizy) - 1] = 5

    if border:
        b = image == 5
    if noiseType == 'norm':
        image = image + noise * np.random.randn(im_size[0], im_size[1])
    elif noiseType == 'uniform':
        image = image + noise * 2 * (np.random.rand(im_size[0], im_size[1]) - .5)
    image[image < 0] = 0
    if border:
        image[image > 1] = 1
        image[b] = 5
    else:
        image[image > 1] = 1
    return (image, rectList, oneObject)


def generate_noise_rect(im_size=[255, 255], sizes=[5, 10, 15],
                        noiseType='norm', prob=None, border=False,
                        sd_lowpass=5):
    """ generates a rectangle image with only noise informaiton
    """
    if prob is None:
        prob = np.ones(len(sizes))
    sizes = np.array(sizes)
    prob = np.array(prob)
    assert np.all(prob > 0), 'probabilities for shapes must be >0'
    assert prob.size == sizes.size, 'probabilities and sizes should have equal length'

    # correction for the different size of the possible area
    if len(np.array(sizes).shape) == 1:
        probx = prob * (sizes+im_size[0]-1)/(np.max(sizes)+im_size[0]-1)
        proby = prob * (sizes+im_size[1]-1)/(np.max(sizes)+im_size[1]-1)
        probx = probx/np.sum(probx)
        proby = proby/np.sum(proby)
        probcx = probx.cumsum()
        probcy = proby.cumsum()
    else:
        prob = prob * (sizes[:, 0] + im_size[0] - 1) / (np.max(sizes[:, 0]) + im_size[0] - 1)
        prob = prob * (sizes[:, 1] + im_size[1] - 1) / (np.max(sizes[:, 1]) + im_size[1] - 1)
        prob = prob/np.sum(prob)
        probc = prob.cumsum()
    image = np.nan * np.zeros(im_size, dtype='float')
    rectList = list()
    while np.any(np.isnan(image)):
        if len(np.array(sizes).shape) == 1:
            idx_sizex = np.searchsorted(probcx, np.random.rand())
            idx_sizey = np.searchsorted(probcy, np.random.rand())
            sizx = sizes[idx_sizex]
            sizy = sizes[idx_sizey]
        elif len(np.array(sizes).shape) == 2:
            idx_size = np.searchsorted(probc, np.random.rand())
            sizx = sizes[idx_size][0]
            sizy = sizes[idx_size][1]
        idx_x = np.random.randint(1 - sizx, im_size[0])
        idx_y = np.random.randint(1 - sizy, im_size[1])
        rectList.append([idx_x, idx_y, sizx, sizy])
        image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)),
              int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = 1
    rectList = np.array(rectList, dtype=np.int16)
    for i in range(len(rectList)):
        noise = 0.25 * np.random.randn(im_size[0], im_size[1])
        noise = 0.5+gaussian_filter(noise, sd_lowpass, mode='wrap')
        image[
            int(max(rectList[len(rectList) - i - 1, 0], 0)):
            int(max(0, rectList[len(rectList) - i - 1, 0] + rectList[len(rectList) - i - 1, 2])),
            int(max(rectList[len(rectList) - i - 1, 1], 0)):
            int(max(0, rectList[len(rectList) - i - 1, 1] + rectList[len(rectList) - i - 1, 3]))]\
            = noise[
                int(max(rectList[len(rectList) - i - 1, 0], 0)):
                int(max(0, rectList[len(rectList) - i - 1, 0]
                        + rectList[len(rectList) - i - 1, 2])),
                int(max(rectList[len(rectList) - i - 1, 1], 0)):
                int(max(0, rectList[len(rectList) - i - 1, 1]
                        + rectList[len(rectList) - i - 1, 3]))]
        if border:
            idx_x = rectList[len(rectList) - i - 1, 0]
            idx_y = rectList[len(rectList) - i - 1, 1]
            sizx = rectList[len(rectList) - i - 1, 2]
            sizy = rectList[len(rectList) - i - 1, 3]
            if idx_x >= 0:
                image[int(idx_x), int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = 5
            if (idx_x+sizx) <= im_size[0]:
                image[int((idx_x+sizx)-1), int(max(idx_y, 0)):int(idx_y + sizy)] = 5
            if idx_y >= 0:
                image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y)] = 5
            if (idx_y+sizy) <= im_size[1]:
                image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y + sizy) - 1] = 5

    if border:
        b = image == 5
    image[image < 0] = 0
    image[image > 1] = 1
    if border:
        image[b] = 5
    return (image, rectList)


def calc_prob_one(sizes=[5, 10, 15], grid=None, prob=None, dx=1, dy=1):
    sizes = np.array(sizes)
    if grid is not None:
        sizes = sizes/grid
        dx = dx/grid
        dy = dy/grid
    if prob is None:
        prob = np.ones(len(sizes))
    if len(sizes.shape) == 1:
        sizes = cartesian([sizes, sizes])
        prob = np.outer(prob, prob).flatten()
    p1 = sum(prob * np.array([max(0, (sizes[k, 0] - dx)) * max(0, (sizes[k, 1] - dy))
                              for k in range(len(sizes))]))
    p2 = sum(prob * np.array([2 * sizes[k, 0] * sizes[k, 1]
                              - 2 * max(0, (sizes[k, 0] - dx))
                              * max(0, (sizes[k, 1] - dy))
                              for k in range(len(sizes))]))
    p = p1 / (p1 + p2)
    return p


def calc_prob_one_grid(sizes=[5, 10, 15], grid=None, prob=None, dx=1, dy=1):
    ps = np.zeros((len(dx), len(dy)))
    kx = 0
    for idx in dx:
        ky = 0
        for idy in dy:
            ps[kx, ky] = calc_prob_one(sizes=sizes, grid=grid, prob=prob, dx=idx, dy=idy)
            ky += 1
        kx += 1
    return ps


def calc_distance_distribution(ps):
    p_diff = np.zeros_like(ps)
    x = np.arange(ps.shape[0], dtype=np.int)
    y = np.arange(ps.shape[1], dtype=np.int)
    yy, xx = np.meshgrid(y, x)
    xx = xx.flatten()
    yy = yy.flatten()
    ps = ps.flatten()
    for i in range(len(ps)):
        xx_not_i = np.concatenate((xx[:i], xx[(i+1):]))
        yy_not_i = np.concatenate((yy[:i], yy[(i+1):]))
        ps_not_i = np.concatenate((ps[:i], ps[(i+1):]))
        ps_not_i = ps_not_i/np.sum(ps_not_i)
        for j in range(len(ps_not_i)):
            x_diff = np.abs(xx_not_i[j] - xx[i])
            y_diff = np.abs(yy_not_i[j] - yy[i])
            p_diff[x_diff, y_diff] += ps[i] * ps_not_i[j]
    return p_diff


def calc_prob_same_from_p(ps, sizes=[5, 10, 15], prob=None, grid=None):
    p_diff = calc_distance_distribution(ps)
    p_same = calc_prob_one_grid(sizes=sizes, prob=prob, grid=grid,
                                dx=np.arange(ps.shape[0]), dy=np.arange(ps.shape[1]))
    return np.sum(p_diff*p_same)


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
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1) * m, 1:] = out[0:m, 1:]
    return out


def fast_rect_conv(im, rect_size):
    # always convolves along the first axis
    imOut = np.zeros(np.array(im.shape) + (rect_size - 1, 0))
    current = np.zeros(im.shape[1])
    for iC in range(im.shape[0] + rect_size - 1):
        if iC < im.shape[0]:
            current = current+im[iC]
        if iC >= rect_size:
            current = current-im[iC-rect_size]
        imOut[iC] = current
    return imOut


def show_test_positions():
    im_size = np.array((300, 300))
    distances = [5, 10, 20, 40, 80]
    distancesd = [4, 7, 14, 28, 57]
    image = np.ones((300, 300, 3))
    xlen = 2
    for iPos in range(5):
        for angle in range(2):
            for abs_angle in range(2):
                if angle:
                    distance = distancesd[iPos]
                else:
                    distance = distances[iPos]

                if angle and not abs_angle:
                    pos = [[-distance / 2, -distance / 2], [distance / 2, distance / 2]]
                elif angle and abs_angle:
                    pos = [[-distance / 2, distance / 2], [distance / 2, -distance / 2]]
                elif not angle and not abs_angle:
                    pos = [[-distance / 2, 0], [distance / 2, 0]]
                elif not angle and abs_angle:
                    pos = [[0, -distance / 2], [0, distance / 2]]
                pos = np.floor(np.array(pos))
                positions = pos
                positions = np.floor(positions)
                positions_im = np.zeros_like(positions)
                positions_im[:, 1] = np.ceil(im_size / 2) + positions[:, 0]
                positions_im[:, 0] = np.ceil(im_size / 2) - positions[:, 1] - 1
                if distance > 20:
                    for ix in range(1, xlen + 1):
                        for ip in range(2):
                            image[np.uint(positions_im[ip, 0] + ix),
                                  np.uint(positions_im[ip, 1] + ix)] = [1, 0, 0]
                            image[np.uint(positions_im[ip, 0] + ix),
                                  np.uint(positions_im[ip, 1] - ix)] = [1, 0, 0]
                            image[np.uint(positions_im[ip, 0] - ix),
                                  np.uint(positions_im[ip, 1] + ix)] = [1, 0, 0]
                            image[np.uint(positions_im[ip, 0] - ix),
                                  np.uint(positions_im[ip, 1] - ix)] = [1, 0, 0]
                else:
                    image[np.asarray(positions_im, dtype=np.int)[:, 0],
                          np.asarray(positions_im, dtype=np.int)[:, 1], :] = [1, 0, 0]
    image[0, :, :] = 0
    image[-1, :, :] = 0
    image[:, 0, :] = 0
    image[:, -1, :] = 0
    return image


class dlMovie:
    def __init__(self, im_size=[255, 255], sizes=[5, 10, 15],
                 colors=[0, 0.5, 1], grid=1, noise=0, noiseType='norm',
                 prob=None, border=False):
        if prob is None:
            self.prob = np.ones(len(sizes))
        else:
            self.prob = prob
        assert (im_size[0] % grid) == 0, 'Image Size not compatible with grid'
        assert (im_size[1] % grid) == 0, 'Image Size not compatible with grid'
        assert np.all(np.array(sizes) % grid == 0), 'Patch sizes not compatible with grid'
        assert noise >= 0, 'noise is the standard deviation and thus should be >=0'
        assert np.all(self.prob > 0), 'probabilities for shapes must be >0'
        assert len(self.prob) == len(sizes), 'probabilities and sizes should have equal length'
        self.im_size = im_size
        self.sizes = np.array(sizes)
        self.colors = colors
        self.grid = grid
        self.noise = noise
        self.noiseType = noiseType
        self.border = border
        self.image = np.nan * np.zeros(im_size, dtype='float')
        self.rectList = np.zeros((0, 5),dtype=np.int16)
        # correction for the different size of the possible area
        if len(np.array(sizes).shape) == 1:
            probx = self.prob * (self.sizes+im_size[0])/(np.max(self.sizes)+im_size[0])
            proby = self.prob * (self.sizes+im_size[1])/(np.max(self.sizes)+im_size[1])
            probx = probx/np.sum(probx)
            proby = proby/np.sum(proby)
            self.probcx = probx.cumsum()
            self.probcy = proby.cumsum()
        else:
            prob = prob * (sizes[:, 0] + im_size[0]) / (np.max(sizes[:, 0]) + im_size[0])
            prob = prob * (sizes[:, 1] + im_size[1]) / (np.max(sizes[:, 1]) + im_size[1])
            prob = prob / np.sum(prob)
            self.probc = prob.cumsum()

    def add_leaf(self):
        if len(self.sizes.shape) == 1:
            idx_sizex = np.searchsorted(self.probcx, np.random.rand())
            idx_sizey = np.searchsorted(self.probcy, np.random.rand())
            sizx = self.sizes[idx_sizex]
            sizy = self.sizes[idx_sizey]
        elif len(self.sizes.shape) == 2:
            idx_size = np.searchsorted(self.probc, np.random.rand())
            sizx = self.sizes[idx_size][0]
            sizy = self.sizes[idx_size][1]
        idx_color = np.random.randint(len(self.colors))
        c = self.colors[idx_color]
        sizx = sizx/self.grid
        sizy = sizy/self.grid
        idx_x = np.random.randint(1 - sizx, self.im_size[0] / self.grid)
        idx_y = np.random.randint(1 - sizy, self.im_size[1] / self.grid)
        self.rectList = np.append(
            self.rectList,
            [[self.grid * idx_x, self.grid * idx_y, self.grid * sizx, self.grid * sizy, idx_color]],
            axis=0)
        self.image[int(self.grid * max(idx_x, 0)):int(self.grid * max(0, idx_x + sizx)),
                   int(self.grid * max(idx_y, 0)):int(self.grid * max(0, idx_y + sizy))] = c
        if self.border:
            if idx_x >= 0:
                self.image[int(idx_x), int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = 5
            if (idx_x + sizx) <= self.im_size[0]:
                self.image[int(idx_x + sizx - 1), int(max(idx_y, 0)):int(idx_y + sizy)] = 5
            if idx_y >= 0:
                self.image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y)] = 5
            if (idx_y+sizy) <= self.im_size[1]:
                self.image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y + sizy) - 1] = 5

    def get_image(self):
        return self.image

    def save_video(self, filename, n_frame):
        out = cv2.VideoWriter()
        out.open(filename, cv2.VideoWriter_fourcc(*"MJPG"), 20, (self.im_size[0], self.im_size[1]))
        for i in range(n_frame):
            im = self.get_image()
            im2 = np.repeat(im.reshape(1, self.im_size[0], self.im_size[1]), 3, 0)
            im2[2][np.isnan(im)] = 0
            im2[1][np.isnan(im)] = 0
            im2[0][np.isnan(im)] = 1
            im2[2][im == 5] = 1
            im2[1][im == 5] = 0.5
            im2[0][im == 5] = 0.5
            im2 = 255 * im2
            im2 = im2.transpose(2, 1, 0)
            im2 = im2.astype('uint8')
            out.write(im2)
            self.add_leaf()
        out.release()


class node:
    def __init__(self):
        self.children = None
        self.probChild = None

    def add_children(self, image, sizes, colors, prob, silent=False):
        self.children = list()
        self.probChild = list()
        self.probInvisibleChild = list()
        kP = 0
        im2 = ~np.isnan(image)
        sizes = np.int32(sizes)
        for iSize in tqdm.tqdm(sizes, disable=silent):
            fieldSize = np.prod(image.shape + iSize - 1)
            # Found better convolution with same result!
            imTest2 = fast_rect_conv(im2, iSize[0])
            imTest2 = fast_rect_conv(imTest2.transpose(), iSize[1]).transpose()
            locationsInvisible = np.where(imTest2 <= 20 * np.finfo(np.float64).eps)
            for t in np.array(locationsInvisible).T:
                self.probInvisibleChild.append(prob[kP]/fieldSize)
            kC = 0
            for iC in colors:
                im = (image - iC) ** 2
                im[np.isnan(im)] = 0
                imTest = fast_rect_conv(im, iSize[0])
                imTest = fast_rect_conv(imTest.transpose(), iSize[1]).transpose()
                locations = np.where(np.logical_and(imTest <= 20 * np.finfo(np.float64).eps,
                                                    imTest2 >= 20 * np.finfo(np.float64).eps))
                for t in np.array(locations).T:
                    self.children.append([t[0] - iSize[0] + 1, t[1] - iSize[1] + 1,
                                          iSize[0], iSize[1], kC, imTest2[t[0], t[1]]])
                    self.probChild.append(prob[kP]/fieldSize/len(colors))
                kC = kC+1
            kP = kP+1
        self.probInvisible = np.sum(np.array(self.probInvisibleChild))
        self.probPossible = np.sum(np.array(self.probChild))

    def get_sample_child(self, image, sizes, colors, prob, silent=False):
        # NOTE: This changes the image although it is not returned!
        if self.children is None:
            self.add_children(image, sizes, colors, prob, silent=silent)
        pc = np.cumsum(self.probChild)
        pc = pc / pc[-1]
        ran = np.random.rand()
        idx = np.argmax(ran < pc)
        child = self.children[idx]
        idx_x = child[0]
        idx_y = child[1]
        sizx = child[2]
        sizy = child[3]
        image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)),
              int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = np.nan
        return (child, self.probPossible, self.probInvisible)

    def get_sample_child_explained_bias(self, image, sizes, colors, prob, silent=False):
        # NOTE: This changes the image although it is not returned!
        if self.children is None:
            self.add_children(image, sizes, colors, prob, silent=silent)
        pCorrection = np.array(self.children)[:, 5] * np.log(len(colors)) + np.log(self.probChild)
        pCorrection = pCorrection-np.max(pCorrection)
        p = np.exp(pCorrection)
        pc = np.cumsum(p)
        p = p/pc[-1]
        pc = pc/pc[-1]
        ran = np.random.rand()
        idx = np.argmax(ran < pc)
        child = self.children[idx]
        idx_x = child[0]
        idx_y = child[1]
        sizx = child[2]
        sizy = child[3]
        image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)),
              int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = np.nan
        logpCorrection = np.log(p[idx]) - np.log(self.probChild[idx]) + np.log(self.probPossible)
        return (child, self.probPossible, self.probInvisible, logpCorrection)

    def get_ML_child(self, image, sizes, colors, prob, silent=False):
        # NOTE: This changes the image although it is not returned!
        if self.children is None:
            self.add_children(image, sizes, colors, prob, silent=silent)
        pCorrection = np.array(self.children)[:, 5] + np.log(self.probChild)
        # pCorrection = pCorrection-np.max(pCorrection)
        idx = np.argmax(pCorrection)
        child = self.children[idx]
        # print(self.children[idx])
        idx_x = child[0]
        idx_y = child[1]
        sizx = child[2]
        sizy = child[3]
        image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)),
              int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = np.nan
        logpCorrection = pCorrection[idx] - np.log(self.probChild[idx])
        return (child, self.probPossible, self.probInvisible, logpCorrection)


class graph:
    def __init__(self, image, sizes=default_sizes, colors=default_colors_255, prob=None):
        image = np.array(image)
        if len(image.shape) > 2:
            image[image[:, :, 0] != image[:, :, 1], :] = np.nan
            image = np.mean(image, axis=2)
        self.image = image
        im_size = self.image.shape
        if prob is None:
            self.prob = np.ones(len(sizes))
        else:
            self.prob = prob
        assert np.all(self.prob > 0), 'probabilities for shapes must be >0'
        assert len(self.prob) == len(sizes), 'probabilities and sizes should have equal length'
        if len(np.array(sizes).shape) == 1:
            self.sizes = np.reshape(np.concatenate(np.meshgrid(sizes, sizes), axis=0),
                                    [2, len(sizes) ** 2]).transpose()
            self.prob = np.outer(self.prob, self.prob).flatten()
        else:
            self.sizes = np.array(sizes)
        self.colors = colors
        self.prob = self.prob * (self.sizes[:, 0] + im_size[0] - 1) \
            / (np.max(self.sizes[:, 0]) + im_size[0] - 1)
        self.prob = self.prob * (self.sizes[:, 1] + im_size[1] - 1) \
            / (np.max(self.sizes[:, 1]) + im_size[1] - 1)
        self.prob = self.prob / np.sum(self.prob)
        self.probc = self.prob.cumsum()

    def get_decomposition(self, points=None, silent=False):
        logPPos = 0
        logPVis = 0
        rectList = np.zeros((0, 6),dtype=np.int16)
        if points is not None:
            points = np.array(points)
        n0 = node()
        im = np.copy(self.image)
        n = n0
        all_contained = None
        k = 0
        while np.any(~np.isnan(im)):
            n = node()
            (rect, pPos, pInVis) = n.get_sample_child(
                im, self.sizes, self.colors, self.prob, silent=silent)
            logPPos = logPPos + np.log(pPos)
            logPVis = logPVis + np.log(1 - pInVis)
            k += 1
            rectList = np.append(rectList, [rect], axis=0)
            if not silent:
                print(k)
                print(np.sum(~np.isnan(im)))
            if all_contained is None and points is not None:
                if np.all(np.logical_and(
                        np.logical_and(points[:, 0] >= rectList[-1, 0],
                                       points[:, 0] < (rectList[-1, 0] + rectList[-1, 2])),
                        np.logical_and(points[:, 1] >= rectList[-1, 1],
                                       points[:, 1] < (rectList[-1, 1] + rectList[-1, 3])))):
                    all_contained = True
                elif np.any(np.logical_and(
                        np.logical_and(points[:, 0] >= rectList[-1, 0],
                                       points[:, 0] < (rectList[-1, 0] + rectList[-1, 2])),
                        np.logical_and(points[:, 1] >= rectList[-1, 1],
                                       points[:, 1] < (rectList[-1, 1] + rectList[-1, 3])))):
                    all_contained = False
        logPCorrection = 0  # - rectList.shape[0]* np.log(len(self.colors))
        return (rectList, all_contained, logPPos, logPVis, logPCorrection)

    def get_decomposition_explained_bias(self, points=None, silent=False):
        logPPos = 0
        logPVis = 0
        logPCorrection = 0
        rectList = np.zeros((0, 6), dtype=np.int16)
        if points is not None:
            points = np.array(points)
        n0 = node()
        im = np.copy(self.image)
        n = n0
        all_contained = None
        k = 0
        while np.any(~np.isnan(im)):
            n = node()
            (rect, pPos, pInVis, correction) = n.get_sample_child_explained_bias(
                im, self.sizes, self.colors, self.prob, silent=silent)
            logPPos = logPPos + np.log(pPos)
            logPVis = logPVis + np.log(1-pInVis)
            logPCorrection = logPCorrection-correction
            k += 1
            rectList = np.append(rectList, [rect], axis=0)
            if not silent:
                print(k)
                print(np.sum(~np.isnan(im)))
            if all_contained is None and points is not None:
                if np.all(np.logical_and(
                        np.logical_and(points[:, 0] >= rectList[-1, 0],
                                       points[:, 0] < (rectList[-1, 0] + rectList[-1, 2])),
                        np.logical_and(points[:, 1] >= rectList[-1, 1],
                                       points[:, 1] < (rectList[-1, 1] + rectList[-1, 3])))):
                    all_contained = True
                elif np.any(np.logical_and(
                        np.logical_and(points[:, 0] >= rectList[-1, 0],
                                       points[:, 0] < (rectList[-1, 0] + rectList[-1, 2])),
                        np.logical_and(points[:, 1] >= rectList[-1, 1],
                                       points[:, 1] < (rectList[-1, 1] + rectList[-1, 3])))):
                    all_contained = False
        return (rectList, all_contained, logPPos, logPVis, logPCorrection)

    def get_decomposition_max_explained(self, points=None, silent=False):
        logPPos = 0
        logPVis = 0
        logPCorrection = 0
        rectList = np.zeros((0, 6), dtype=np.int16)
        if points is not None:
            points = np.array(points)
        n0 = node()
        im = np.copy(self.image)
        n = n0
        all_contained = None
        k = 0
        while np.any(~np.isnan(im)):
            n = node()
            (rect, pPos, pInVis, correction) = n.get_ML_child(
                im, self.sizes, self.colors, self.prob, silent=silent)
            logPPos = logPPos + np.log(pPos)
            logPVis = logPVis + np.log(1 - pInVis)
            logPCorrection = logPCorrection-correction
            k += 1
            rectList = np.append(rectList, [rect], axis=0)
            if not silent:
                print(k)
                print(np.sum(~np.isnan(im)))
            if all_contained is None and points is not None:
                if np.all(np.logical_and(
                        np.logical_and(points[:, 0] >= rectList[-1, 0],
                                       points[:, 0] < (rectList[-1, 0] + rectList[-1, 2])),
                        np.logical_and(points[:, 1] >= rectList[-1, 1],
                                       points[:, 1] < (rectList[-1, 1] + rectList[-1, 3])))):
                    all_contained = True
                elif np.any(np.logical_and(
                        np.logical_and(points[:, 0] >= rectList[-1, 0],
                                       points[:, 0] < (rectList[-1, 0] + rectList[-1, 2])),
                        np.logical_and(points[:, 1] >= rectList[-1, 1],
                                       points[:, 1] < (rectList[-1, 1] + rectList[-1, 3])))):
                    all_contained = False
        return (rectList, all_contained, logPPos, logPVis, logPCorrection)

    def get_exact_prob(self, points, silent=False):
        if not silent:
            np.set_printoptions(precision=2, linewidth=100, floatmode='fixed')
        points = np.array(points)
        n0 = node()
        im = np.copy(self.image)
        n0.add_children(im, self.sizes, self.colors, self.prob, silent=True)
        nodes = [n0]
        images = [im]
        p_node = 0
        p_nodes = [p_node]
        p_node_same = 0
        p_same = [p_node_same]
        p_prior = 1
        p_priors = [p_prior]
        same_rect_node = None
        same_rect = [same_rect_node]
        while len(nodes) > 0:
            n = nodes[-1]
            im = images[-1]
            if not silent:
                print(np.array(p_nodes), end="\r")
            if len(n.children) > 0:
                rect = n.children.pop()
                p_child = n.probChild.pop()
                p_invis = n.probInvisible
                same_rect_node = same_rect[-1]
                n_new = node()
                im_new = np.copy(im)
                idx_x = rect[0]
                idx_y = rect[1]
                sizx = rect[2]
                sizy = rect[3]
                im_new[int(max(idx_x, 0)):int(max(0, idx_x + sizx)),
                       int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = np.nan
                if same_rect_node is None:
                    if np.all((idx_x <= points[:, 0]) & (idx_x + sizx > points[:, 0])
                              & (idx_y <= points[:, 1]) & (idx_y + sizy > points[:, 1])):
                        same_rect_node = True
                    elif np.any((idx_x <= points[:, 0]) & (idx_x + sizx > points[:, 0])
                                & (idx_y <= points[:, 1]) & (idx_y + sizy > points[:, 1])):
                        same_rect_node = False
                same_rect.append(same_rect_node)
                n_new.add_children(im_new, self.sizes, self.colors, self.prob, silent=True)
                p_prior_new = p_priors[-1] * p_child / (1 - p_invis)
                p_node_same = 0
                p_same.append(p_node_same)
                p_node = 0
                p_nodes.append(p_node)
                p_priors.append(p_prior_new)
                nodes.append(n_new)
                images.append(im_new)
            else:
                im = images.pop()
                n = nodes.pop()
                p_prior = p_priors.pop()
                p_node = p_nodes.pop()
                p_node_same = p_same.pop()
                same_rect_node = same_rect.pop()
                if len(nodes) > 0:
                    if np.all(np.isnan(im)):
                        p_nodes[-1] = p_nodes[-1] + p_prior
                        if same_rect_node:
                            p_same[-1] = p_same[-1] + p_prior
                    else:
                        p_nodes[-1] = p_nodes[-1] + p_node
                        p_same[-1] = p_same[-1] + p_node_same
        return p_node_same, p_node


def mark_points(image, positions_im):
    image[:, :, :1] = 0
    image[np.asarray(positions_im, dtype=np.int)[:, 0],
          np.asarray(positions_im, dtype=np.int)[:, 1], 0] = 1
    return image


def generate_image(distance, angle, abs_angle,
                   sizes=np.array([5, 10, 15, 20, 25, 30]),
                   exponent=3,
                   border=False,
                   im_size=np.array([30, 30]),
                   num_colors=9,
                   m_points=True,
                   same_color=0):
    """
    Parameters
    ----------
    distance : float
    angle : float
    abs_angle : float
    exponent : float, optional
        DESCRIPTION. The default is 3.
    border : TYPE, optional
        DESCRIPTION. The default is False.
    sizes : TYPE, optional
        DESCRIPTION. The default is np.array([5,10,15,20,25,30]).
    im_size : TYPE, optional
        DESCRIPTION. The default is np.array([30,30]).
    num_colors : TYPE, optional
        number of colors used. The default is 9.
    m_points : TYPE, optional
        whether points are marked. The default is True.
    same_color : TYPE, optional
        how to force the two points to have the same color.
        0(default) -> not at all
        1(old)     -> color in rectangles
        2(new)     -> generate images until it is true

    Returns
    -------
    image : np.ndarray
        DESCRIPTION.
    rect_list :
        true rectangles
    positions_im : np.ndarray
        Querried points.
    solution : bool
        are the two points on the same rectangle?
    col : int
        which color did the points have?

    """
    prob = (sizes / np.min(sizes)) ** -(exponent / 2)
    if angle and not abs_angle:
        pos = [[-distance / 2, -distance / 2], [distance / 2, distance / 2]]
    elif angle and abs_angle:
        pos = [[-distance / 2, distance / 2], [distance / 2, -distance / 2]]
    elif not angle and not abs_angle:
        pos = [[-distance / 2, 0], [distance / 2, 0]]
    elif not angle and abs_angle:
        pos = [[0, - distance / 2], [0, distance / 2]]
    pos = np.floor(np.array(pos))

    positions = pos
    positions = np.floor(positions)
    positions_im = np.zeros_like(positions, dtype=np.int)
    positions_im[:, 1] = np.ceil(im_size / 2) + positions[:, 0]
    positions_im[:, 0] = np.ceil(im_size / 2) - positions[:, 1] - 1
    if same_color == 0:
        im = gen_rect_leaf(
            im_size,
            sizes=sizes,
            prob=prob,
            grid=1,
            colors=np.linspace(0, 1, num_colors),
            border=border)
        col = im[0][positions_im[0][0], positions_im[0][1]]
    elif same_color == 1:
        col = np.random.randint(num_colors)
        im = gen_rect_leaf(
            im_size,
            sizes=sizes,
            prob=prob,
            grid=1,
            colors=np.linspace(0, 1, num_colors),
            fixedIdx=positions_im,
            fixedC=col,
            border=border)
    elif same_color == 2:
        repeat = True
        while repeat:
            im = gen_rect_leaf(
                im_size,
                sizes=sizes,
                prob=prob,
                grid=1,
                colors=np.linspace(0, 1, num_colors),
                border=border)
            if (im[0][positions_im[0][0], positions_im[0][1]] ==
                im[0][positions_im[1][0], positions_im[1][1]]):
                repeat = False
        col = im[0][positions_im[0]]
    solution = test_positions(im[1], positions_im)
    image = im[0]
    image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
    image[im[0] == 5, :] = [.5, .5, 1]
    if m_points:
        image = mark_points(image, positions_im)
    return (image, im[1], positions_im, solution, col)


def generate_image_point(point_probabilities=None,
                         exponent=3,
                         border=False,
                         sizes=np.array([5, 10, 15, 20, 25, 30]),
                         im_size=np.array([30, 30]),
                         num_colors=9,
                         m_points=True,
                         same_color=0):
    """

    Parameters
    ----------
    point_probabilities : np.ndarray
        DESCRIPTION.
    exponent : TYPE, optional
        DESCRIPTION. The default is 3.
    border : TYPE, optional
        DESCRIPTION. The default is False.
    sizes : TYPE, optional
        DESCRIPTION. The default is np.array([5,10,15,20,25,30]).
    im_size : TYPE, optional
        DESCRIPTION. The default is np.array([30,30]).
    num_colors : TYPE, optional
        number of colors used. The default is 9.
    m_points : TYPE, optional
        whether points are marked. The default is True.
    same_color : TYPE, optional
        how to force the two points to have the same color.
        0(default) -> not at all
        1(old)     -> color in rectangles
        2(new)     -> generate images until it is true

    Returns
    -------
    image : np.ndarray
        DESCRIPTION.
    rect_list :
        true rectangles
    positions_im : np.ndarray
        Querried points.
    solution : bool
        are the two points on the same rectangle?
    col : int
        which color did the points have?

    """
    prob = (sizes/np.min(sizes)) ** -(exponent/2)
    if point_probabilities is None:
        point_probabilities = np.ones(im_size)
    else:
        point_probabilities = point_probabilities.copy()
    point_probabilities = (point_probabilities / np.sum(point_probabilities)).flatten()
    r1 = np.random.rand()
    idx1 = np.where(np.cumsum(point_probabilities) > r1)[0][0]
    point_probabilities[idx1] = 0
    point_probabilities = point_probabilities / np.sum(point_probabilities)
    r2 = np.random.rand()
    idx2 = np.where(np.cumsum(point_probabilities) > r2)[0][0]
    positions_im = np.unravel_index((idx1, idx2), im_size)
    positions_im = np.array(positions_im).T
    if same_color == 0:
        im = gen_rect_leaf(
            im_size,
            sizes=sizes,
            prob=prob,
            grid=1,
            colors=np.linspace(0, 1, num_colors),
            border=border)
        col = im[0][positions_im[0]]
    elif same_color == 1:
        col = np.random.randint(num_colors)
        im = gen_rect_leaf(
            im_size,
            sizes=sizes,
            prob=prob,
            grid=1,
            colors=np.linspace(0, 1, num_colors),
            fixedIdx=positions_im,
            fixedC=col,
            border=border)
    elif same_color == 2:
        repeat = True
        while repeat:
            im = gen_rect_leaf(
                im_size,
                sizes=sizes,
                prob=prob,
                grid=1,
                colors=np.linspace(0, 1, num_colors),
                border=border)
            if (im[0][positions_im[0][0], positions_im[0][1]] ==
                im[0][positions_im[1][0], positions_im[1][1]]):
                repeat = False
        col = im[0][positions_im[0]]
    solution = test_positions(im[1], positions_im)
    image = im[0]
    image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
    image[im[0] == 5, :] = [.5, .5, 1]
    if m_points:
        image = mark_points(image, positions_im)
    return (image, im[1], positions_im, solution, col)


def generate_image_dist(dist_probabilities,
                        exponent=3,
                        border=False,
                        sizes=np.array([5, 10, 15, 20, 25, 30]),
                        im_size=np.array([30, 30]),
                        num_colors=9,
                        m_points=True,
                        same_color=0
                        ):
    """

    Parameters
    ----------
    point_probabilities : np.ndarray
        DESCRIPTION.
    exponent : TYPE, optional
        DESCRIPTION. The default is 3.
    border : TYPE, optional
        DESCRIPTION. The default is False.
    sizes : TYPE, optional
        DESCRIPTION. The default is np.array([5,10,15,20,25,30]).
    im_size : TYPE, optional
        DESCRIPTION. The default is np.array([30,30]).
    num_colors : TYPE, optional
        number of colors used. The default is 9.
    m_points : TYPE, optional
        whether points are marked. The default is True.
    same_color : TYPE, optional
        how to force the two points to have the same color.
        0(default) -> not at all
        1(old)     -> color in rectangles
        2(new)     -> generate images until it is true

    Returns
    -------
    image : np.ndarray
        DESCRIPTION.
    rect_list :
        true rectangles
    positions_im : np.ndarray
        Querried points.
    solution : bool
        are the two points on the same rectangle?
    col : int
        which color did the points have?

    """
    im_size = np.array(im_size)
    prob = (sizes / np.min(sizes)) ** -(exponent / 2)
    if dist_probabilities is None:
        dist_probabilities = np.ones(im_size)
    dist_probabilities = (dist_probabilities / np.sum(dist_probabilities)).flatten()
    r = np.random.rand()
    idx = np.where(np.cumsum(dist_probabilities) > r)[0][0]
    dx, dy = np.unravel_index(idx, im_size)
    select_size = im_size - np.array((dx, dy))
    x = np.random.randint(select_size[0])
    y = np.random.randint(select_size[1])
    if np.random.rand() > 0.5:
        positions_im = np.array([[x, y], [x + dx, y + dy]])
    else:
        positions_im = np.array([[x + dx, y], [x, y + dy]])
    if same_color == 0:
        im = gen_rect_leaf(
            im_size,
            sizes=sizes,
            prob=prob,
            grid=1,
            colors=np.linspace(0, 1, num_colors),
            border=border)
        col = im[0][positions_im[0]]
    elif same_color == 1:
        col = np.random.randint(num_colors)
        im = gen_rect_leaf(
            im_size,
            sizes=sizes,
            prob=prob,
            grid=1,
            colors=np.linspace(0, 1, num_colors),
            fixedIdx=positions_im,
            fixedC=col,
            border=border)
    elif same_color == 2:
        repeat = True
        while repeat:
            im = gen_rect_leaf(
                im_size,
                sizes=sizes,
                prob=prob,
                grid=1,
                colors=np.linspace(0, 1, num_colors),
                border=border)
            if (im[0][positions_im[0][0], positions_im[0][1]] ==
                im[0][positions_im[1][0], positions_im[1][1]]):
                repeat = False
        col = im[0][positions_im[0]]
    solution = test_positions(im[1], positions_im)
    image = im[0]
    image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
    image[im[0] == 5, :] = [.5, .5, 1]
    if m_points:
        image = mark_points(image, positions_im)
    return (image, im[1], positions_im, solution, col)


def generate_image_from_rects(im_size, rectList, border=False, colors=None):
    if colors is None:
        colors = np.arange(np.max(rectList[:, 4]) + 1) / np.max(rectList[:, 4])
    image = np.zeros(im_size)
    for i in range(len(rectList)):
        image[int(max(rectList[len(rectList) - i - 1, 0], 0)):
              int(max(0, rectList[len(rectList) - i - 1, 0] + rectList[len(rectList) - i - 1, 2])),
              int(max(rectList[len(rectList) - i - 1, 1], 0)):
              int(max(0, rectList[len(rectList) - i - 1, 1] + rectList[len(rectList) - i - 1, 3]))] \
            = colors[int(rectList[len(rectList) - i - 1, 4])]
        if border:
            idx_x = rectList[len(rectList) - i - 1, 0]
            idx_y = rectList[len(rectList) - i - 1, 1]
            sizx = rectList[len(rectList) - i - 1, 2]
            sizy = rectList[len(rectList) - i - 1, 3]
            if idx_x >= 0:
                image[int(idx_x), int(max(idx_y, 0)):int(max(0, idx_y + sizy))] = 5
            if (idx_x + sizx) <= im_size[0]:
                image[int((idx_x + sizx) - 1), int(max(idx_y, 0)):int(idx_y + sizy)] = 5
            if idx_y >= 0:
                image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y)] = 5
            if (idx_y + sizy) <= im_size[1]:
                image[int(max(idx_x, 0)):int(max(0, idx_x + sizx)), int(idx_y + sizy) - 1] = 5
    image3 = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
    image3[image == 5, :] = [.5, .5, 1]
    return image3


def test_positions(rectList, fixedIdx):
    # find whether two points in fixedIdx are on one object or not
    oneObject = False
    idxStart = -1
    while True:
        idxStart = idxStart+1
        R = rectList[idxStart]
        delete = []
        for i in range(len(fixedIdx)):
            if ((R[0] <= fixedIdx[i][0]) and ((R[0] + R[2]) > fixedIdx[i][0])
                and (R[1] <= fixedIdx[i][1]) and ((R[1] + R[3]) > fixedIdx[i][1])):
                delete.append(i)
        if len(delete) > 1:
            oneObject = True
        if len(delete) > 0:
            break
    return oneObject


def show_frozen_image(
        im_folder='imagesFrozen/', exponent=1, num_colors=9, dist=40,
        angle=0, abs_angle=0, i=0, border=0):
    import PIL
    im_name = im_folder + "image%d_%d_%d_%d_%d_%d_%d.png" % (
        exponent, num_colors, dist, angle, abs_angle, i, border)
    im = PIL.Image.open(im_name)
    im.show()
    return im


def create_training_data(
        N, exponents=np.arange(1, 6), sizes=5 * np.arange(1, 80, dtype='float'),
        im_size=np.array([300, 300]), distances=np.array([5, 10, 20, 40, 80]),
        distancesd=np.array([4, 7, 14, 28, 57]), m_points=True):
    # creates training images on the fly
    images = np.zeros((N, im_size[0], im_size[1], 3))
    solution = np.zeros((N))
    for i in range(N):
        exponent = exponents[np.random.randint(len(exponents))]
        if distances is None:
            distance = None
            angle = None
            abs_angle = None
        else:
            angle = np.random.randint(2)
            abs_angle = np.random.randint(2)
            if angle:
                distance = distancesd[np.random.randint(len(distancesd))]
            else:
                distance = distances[np.random.randint(len(distances))]
        im = generate_image(exponent, 0, sizes, distance, angle, abs_angle,
                            m_points=m_points, im_size=im_size)
        images[i] = im[0]
        if im[3]:
            solution[i] = 1
        else:
            solution[i] = 0
    return images, solution


def save_training_data(root_dir, N, exponents=np.arange(1, 6),
                       sizes=5*np.arange(1, 80,dtype='float'),
                       im_size=np.array([300, 300]),
                       distances=None,
                       distancesd=None,
                       m_points=True,
                       point_probabilities=None,
                       dist_probabilities=None,
                       same_color=0):
    # saves training images into a folder
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    solution_list = list()
    exponent_list = list()
    angle_list = list()
    abs_angle_list = list()
    distance_list = list()
    im_name_list = list()
    pos_x1_list = list()
    pos_x2_list = list()
    pos_y1_list = list()
    pos_y2_list = list()
    for i in tqdm.trange(N, smoothing=0):
        exponent = exponents[np.random.randint(len(exponents))]
        if point_probabilities is not None:
            distance = None
            angle = None
            abs_angle = None
            im = generate_image_point(exponent=exponent, border=False, sizes=sizes,
                                      m_points=m_points,
                                      im_size=im_size,
                                      point_probabilities=point_probabilities,
                                      same_color=same_color)
        elif dist_probabilities is not None:
            distance = None
            angle = None
            abs_angle = None
            im = generate_image_dist(
                exponent=exponent, border=False, sizes=sizes,
                m_points=m_points,
                im_size=im_size,
                dist_probabilities=dist_probabilities,
                same_color=same_color)
        elif distances is not None:
            angle = np.random.randint(2)
            abs_angle = np.random.randint(2)
            if angle:
                distance = distancesd[np.random.randint(len(distancesd))]
            else:
                distance = distances[np.random.randint(len(distances))]
            im = generate_image(exponent, 0, sizes,
                                distance, angle, abs_angle,
                                m_points=m_points,
                                im_size=im_size,
                                same_color=same_color)
        else:
            raise ValueError('You have to specify the point distributions somehow!')
        image = im[0]
        if im[3]:
            solution_list.append(1)
        else:
            solution_list.append(0)
        exponent_list.append(exponent)
        angle_list.append(angle)
        abs_angle_list.append(abs_angle)
        distance_list.append(distance)
        pos_x1_list.append(im[2][0][0])
        pos_x2_list.append(im[2][1][0])
        pos_y1_list.append(im[2][0][1])
        pos_y2_list.append(im[2][1][1])
        im_name = 'image%07d.png' % i
        io.imsave(os.path.join(root_dir, im_name), (255 * image).astype('uint8'))
        im_name_list.append(im_name)
    df = pd.DataFrame({'im_name': im_name_list, 'solution': solution_list,
                       'exponent': exponent_list, 'angle': angle_list,
                       'abs_angle': abs_angle_list, 'distance': distance_list,
                       'pos_x1': pos_x1_list, 'pos_x2': pos_x2_list,
                       'pos_y1': pos_y1_list, 'pos_y2': pos_y2_list})
    df.to_csv(os.path.join(root_dir, 'solution.csv'))


def reduce_image(image, points=None, method='line'):
    image = image.copy().astype(np.float)
    if len(image.shape) == 3:
        image[(image[:, :, 0] == 255) & (image[:, :, 1] == 0), :] = np.nan
        image = np.mean(image, axis=2)
    if points is None:
        if np.any(np.isnan(image)):
            all_idx = np.array(np.where(np.isnan(image)))
            uni = np.unique(all_idx[0], return_index=True, return_counts=True)
            points = all_idx[:, uni[1][uni[2] == 1]].T
            if points.size == 0:
                uni = np.unique(all_idx[1], return_index=True, return_counts=True)
                points = all_idx[:, uni[1][uni[2] == 1]].T
        else:
            raise ValueError('If the image is not marked you have to specify the querried points!')
    assert np.all(points.shape == np.array([2, 2]))
    if method == 'point':
        image[points[:, 0], points[:, 1]] = True
    elif method == 'line':
        mask = np.zeros_like(image, dtype=np.bool)
        if points[0, 0] == points[1, 0]:  # only change in y direction
            mask[points[0, 0], np.arange(points[0, 1], points[1, 1])] = True
        elif points[0, 1] == points[1, 1]:  # only change in x direction
            mask[np.arange(points[0, 0], points[1, 0]), points[0, 1]] = True
        elif points[0, 1] < points[1, 1]:  # diagonal positive
            mask[np.arange(points[0, 0], points[1, 0]),
                 np.arange(points[0, 1], points[1, 1])] = True
        elif points[0, 1] > points[1, 1]:  # diagonal positive
            mask[np.arange(points[0, 0], points[1, 0], -1),
                 np.arange(points[0, 1], points[1, 1], -1)] = True
    elif method == 'square':
        mask = np.zeros_like(image, dtype=np.bool)
        mask[np.min(points[:, 0]):np.max(points[:, 0]),
             np.min(points[:, 1]):np.max(points[:, 1])] = True
    image[~mask] = np.nan
    return image
