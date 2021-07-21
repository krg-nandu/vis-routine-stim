#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:45:25 2019

@author: heiko
Neural network training pytorch variant
"""
import sys
import getopt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import tqdm
import matplotlib.pyplot as plt
import pathlib

import DeadLeaf as dl

# for data loading
import pandas as pd
from skimage import io  # transform
from torch.utils.data import Dataset, DataLoader

eps = 10**-5

# standard parameters

sizes = np.arange(1, 6, dtype=np.float)
imSize = np.array([5, 5])


# Model definitions

def init_weights_layer_conv(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


def init_weights_layer_linear(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight,
                gain=0.1 * nn.init.calculate_gain('relu'))
        layer.bias.data.fill_(0)


def activity_one_regularizer(activities):
    return torch.mean((activities-1) * (activities-1))


class model_min_class(nn.Module):
    def __init__(self):
        super(model_min_class, self).__init__()
        self.fc = nn.Linear(3 * imSize[0] * imSize[1], 1)

    def forward(self, x):
        x = x.view(-1, 3 * imSize[0] * imSize[1])
        x = self.fc(x)
        return x

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_class(nn.Module):
    def __init__(self, im_size):
        super(model_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(3*im_size*im_size, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
        self.im_size = im_size

    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 3 * self.im_size * self.im_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_residual(nn.Module):
    def __init__(self, im_size):
        super(model_residual, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, padding=(2, 2))
        self.conv1_1 = nn.Conv2d(20, 20, kernel_size=5, padding=(2, 2))
        self.conv1_2 = nn.Conv2d(20, 20, kernel_size=5, padding=(2, 2))
        self.conv1_3 = nn.Conv2d(20, 20, kernel_size=5, padding=(2, 2))
        self.conv1_4 = nn.Conv2d(20, 20, kernel_size=5, padding=(2, 2))
        self.conv1_5 = nn.Conv2d(20, 20, kernel_size=5, padding=(2, 2))
        self.conv1_6 = nn.Conv2d(20, 20, kernel_size=5, padding=(2, 2))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, padding=(1, 1))
        self.conv2_1 = nn.Conv2d(40, 40, kernel_size=3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(40, 40, kernel_size=3, padding=(1, 1))
        self.conv2_3 = nn.Conv2d(40, 40, kernel_size=3, padding=(1, 1))
        self.conv2_4 = nn.Conv2d(40, 40, kernel_size=3, padding=(1, 1))
        self.conv2_5 = nn.Conv2d(40, 40, kernel_size=3, padding=(1, 1))
        self.conv2_6 = nn.Conv2d(40, 40, kernel_size=3, padding=(1, 1))
        self.fc1 = nn.Linear(40*np.prod(np.floor(im_size/2)), 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
        self.im_size = im_size

    def forward(self, x):
        x = self.norm(x)
        x1 = F.relu(self.conv1(x))
        xres = F.relu(self.conv1_1(x1))
        xres = F.relu(self.conv1_2(xres))
        x1_1 = self.conv1_3(xres) + x1
        xres = F.relu(self.conv1_4(x1_1))
        xres = F.relu(self.conv1_5(xres))
        x1_2 = self.conv1_6(xres) + x1_1
        x2 = self.conv2(self.pool(x1_2))
        xres = F.relu(self.conv2_1(x2))
        xres = F.relu(self.conv2_2(xres))
        x2_1 = self.conv2_3(xres) + x2
        xres = F.relu(self.conv2_4(x2_1))
        xres = F.relu(self.conv2_5(xres))
        x2_2 = self.conv2_6(xres) + x2_1
        x = x2_2.view(-1, 40*np.floor(self.im_size/2)**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model2_class(nn.Module):
    def __init__(self, im_size):
        super(model2_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(3*im_size**2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)
        self.im_size = im_size

    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 3 * self.im_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_BLT(nn.Module):
    def __init__(self, im_size, n_rep=5, n_neurons=10, kernel=3, L=True,
                 T=True, average=False):
        super(model_BLT, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        # Bottom-up
        self.conv0_1 = nn.Conv2d(3, n_neurons, kernel_size=kernel,
                                 padding=(int((kernel-1)/2),
                                          int((kernel-1)/2)))
        self.conv1_2 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel,
                                 padding=(int((kernel-1)/2),
                                          int((kernel-1)/2)))
        self.conv2_3 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel,
                                 padding=(int((kernel-1)/2),
                                          int((kernel-1)/2)))
        if L:
            # Lateral
            self.conv1_1 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel,
                                     padding=(int((kernel-1)/2),
                                              int((kernel-1)/2)), bias=False)
            self.conv2_2 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel,
                                     padding=(int((kernel-1)/2),
                                              int((kernel-1)/2)), bias=False)
            self.conv3_3 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel,
                                     padding=(int((kernel-1)/2),
                                              int((kernel-1)/2)), bias=False)
        if T:
            # top-down
            self.conv2_1 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel,
                                     padding=(int((kernel-1)/2),
                                              int((kernel-1) / 2)), bias=False)
            self.conv3_2 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel,
                                     padding=(int((kernel-1)/2),
                                              int((kernel-1) / 2)), bias=False)
        if not average:
            self.fc1 = nn.Linear(n_neurons*im_size*im_size, n_neurons)
        self.fc2 = nn.Linear(n_neurons, 1)
        self.n_rep = n_rep
        self.n_neurons = n_neurons
        self.L = L
        self.T = T
        self.im_size = im_size
        self.average = average

    def forward(self, x):
        x = self.norm(x)
        im_size = self.im_size
        h1 = torch.zeros([x.shape[0], self.n_neurons, im_size, im_size],
                         device=x.device)
        h2 = torch.zeros([x.shape[0], self.n_neurons, im_size, im_size],
                         device=x.device)
        h3 = torch.zeros([x.shape[0], self.n_neurons, im_size, im_size],
                         device=x.device)
        for i in range(self.n_rep):
            if self.L and self.T:
                h1 = F.relu(self.conv0_1(x)+self.conv1_1(h1)+self.conv2_1(h2))
                h2 = F.relu(self.conv1_2(h1)+self.conv2_2(h2)+self.conv3_2(h3))
                h3 = F.relu(self.conv2_3(h2)+self.conv3_3(h3))
            elif self.L:
                h1 = F.relu(self.conv0_1(x)+self.conv1_1(h1))
                h2 = F.relu(self.conv1_2(h1)+self.conv2_2(h2))
                h3 = F.relu(self.conv2_3(h2)+self.conv3_3(h3))
            elif self.T:
                h1 = F.relu(self.conv0_1(x)+self.conv2_1(h2))
                h2 = F.relu(self.conv1_2(h1)+self.conv3_2(h3))
                h3 = F.relu(self.conv2_3(h2))
            else:
                h1 = F.relu(self.conv0_1(x))
                h2 = F.relu(self.conv1_2(h1))
                h3 = F.relu(self.conv2_3(h2))
        if self.average:
            x = torch.mean(torch.mean(h3, 3), 2)
        else:
            x = h3.view(-1, self.n_neurons * im_size * im_size)
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_recurrent(nn.Module):
    def __init__(self, im_size, n_rep=20, n_neurons=10000):
        super(model_recurrent, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(n_neurons+(3*im_size*im_size), n_neurons)
        self.fc2 = nn.Linear(n_neurons, 1)
        self.n_rep = n_rep
        self.n_neurons = n_neurons
        self.im_size = im_size

    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 3*self.im_size * self.im_size)
        siz = list(x.shape)
        siz[1] = self.n_neurons
        h1 = torch.ones(siz, device=x.device)
        for i in range(self.n_rep):
            inp1 = torch.cat((x,
                              (h1 - torch.mean(h1, dim=1).view(-1, 1))
                              / (eps + h1.std())),
                             dim=1)
            h1 = F.relu(self.fc1(inp1))
        x = self.fc2(h1)
        return x

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


def get_shifted_values(feat, neighbors):
    out = []
    for iNeigh in range(neighbors.shape[0]):
        output = torch.zeros(feat.shape)
        if neighbors[iNeigh, 0] >= 0 and neighbors[iNeigh, 1] >= 0:
            output[:, :, :(feat.shape[2]-int(neighbors[iNeigh, 0])),
                   :(feat.shape[3] - int(neighbors[iNeigh, 1]))] = \
                       feat[:, :, int(neighbors[iNeigh, 0]):,
                            int(neighbors[iNeigh, 1]):]
        elif neighbors[iNeigh, 0] >= 0 and neighbors[iNeigh, 1] < 0:
            output[:, :, :(feat.shape[2] - int(neighbors[iNeigh, 0])),
                   int(-neighbors[iNeigh, 1]):] = \
                       feat[:, :, int(neighbors[iNeigh, 0]):,
                            :(feat.shape[3] - int(-neighbors[iNeigh, 1]))]
        elif neighbors[iNeigh, 0] < 0 and neighbors[iNeigh, 1] >= 0:
            output[:, :, int(-neighbors[iNeigh, 0]):,
                   :(feat.shape[3] - int(neighbors[iNeigh, 1]))] = \
                       feat[:, :,
                            :(feat.shape[2] - int(-neighbors[iNeigh, 0])),
                            int(neighbors[iNeigh, 1]):]
        elif neighbors[iNeigh, 0] < 0 and neighbors[iNeigh, 1] < 0:
            output[:, :, int(-neighbors[iNeigh, 0]):,
                   int(-neighbors[iNeigh, 1]):] = \
                       feat[:, :,
                            :(feat.shape[2] - int(-neighbors[iNeigh, 0])),
                            :(feat.shape[3] - int(-neighbors[iNeigh, 1]))]
        out.append(output)
    output = torch.cat(out, 0).reshape(neighbors.shape[0],
                                       feat.shape[0],
                                       feat.shape[1],
                                       feat.shape[2],
                                       feat.shape[3])
    return output


class model_pred_like(nn.Module):
    def __init__(self, n_rep=5,
                 neighbors=np.array([[0, -1], [0, 1], [1, 0], [-1, 0]])):
        super(model_pred_like, self).__init__()
        self.neighbors = neighbors
        self.norm = nn.InstanceNorm2d(3)
        self.conv1value = nn.Conv2d(3, 10, kernel_size=5,
                                    stride=1, padding=(2, 2))
        self.conv1prec = nn.Conv2d(3, 10, kernel_size=5,
                                   stride=1, padding=(2, 2))
        self.conv1neigh = nn.Conv2d(3, 4, kernel_size=5,
                                    stride=1, padding=(2, 2))
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2value = nn.Conv2d(10, 10, kernel_size=3,
                                    stride=1, padding=(1, 1))
        self.conv2prec = nn.Conv2d(10, 10, kernel_size=3,
                                   stride=1, padding=(1, 1))
        self.conv2neigh = nn.Conv2d(10, 4, kernel_size=3,
                                    stride=1, padding=(1, 1))
        self.pool2 = nn.MaxPool2d(6, 6)
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
        self.logC1 = torch.nn.Parameter(torch.Tensor(
                -2*np.ones((self.neighbors.shape[0], 10))))
        self.register_parameter('logC_1', self.logC1)
        self.logC2 = torch.nn.Parameter(torch.Tensor(
                -2*np.ones((self.neighbors.shape[0], 10))))
        self.register_parameter('logC_2', self.logC2)
        self.n_rep = n_rep

    def forward(self, x):
        x = self.norm(x)
        epsilon = 0.000001
        value1in = F.relu(self.conv1value(x))
        prec1in = F.relu(self.conv1prec(x)) + epsilon
        neigh1 = F.relu(self.conv1neigh(x))
        neigh1 = neigh1.unsqueeze(0).permute((2, 1, 0, 3, 4))
        value1 = value1in
        prec1 = prec1in
        out1 = self.pool(value1)
        value2in = F.relu(self.conv2value(out1))
        prec2in = F.relu(self.conv2prec(out1)) + epsilon
        neigh2 = F.relu(self.conv2neigh(out1))
        neigh2 = neigh2.unsqueeze(0).permute((2, 1, 0, 3, 4))
        value2 = value2in
        prec2 = prec2in
        neighbors = self.neighbors
        C1 = torch.exp(self.logC1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        C2 = torch.exp(self.logC2).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        for i in range(self.n_rep):
            neighValues1 = get_shifted_values(value1, neighbors)
            neighPrec1 = get_shifted_values(prec1, neighbors)
            neighPrec1 = ((neigh1 * C1 * neighPrec1)
                          / (neigh1 * C1 + neighPrec1 + epsilon))
            prec1 = prec1in + torch.sum(neighPrec1, 0)
            value1 = ((prec1in * value1in
                       + torch.sum(neighPrec1 * neighValues1, 0))
                      / prec1)
            out1 = self.pool(value1)
            value2in = F.relu(self.conv2value(out1))
            prec2in = F.relu(self.conv2prec(out1)) + epsilon
            neigh2 = F.relu(self.conv2neigh(out1))
            neigh2 = neigh2.unsqueeze(0).permute((2, 1, 0, 3, 4))
            neighValues2 = get_shifted_values(value2, neighbors)
            neighPrec2 = get_shifted_values(prec2, neighbors)
            neighPrec2 = ((neigh2 * C2 * neighPrec2)
                          / (neigh2 * C2 + neighPrec2 + epsilon))
            prec2 = prec2in + torch.sum(neighPrec2, 0)
            value2 = ((prec2in * value2in
                       + torch.sum(neighPrec2 * neighValues2, 0))
                      / prec2)
            out2 = self.pool(value2)
        x = out2.view(-1, 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


# Dataset definition (for reading from disk)
class dead_leaves_dataset(Dataset):
    """dead leaves dataset."""

    def __init__(self, root_dir, transform=None, n_data=None, repeats=1):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.solutions_df = pd.read_csv(os.path.join(root_dir, 'solution.csv'),
                                        index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        if n_data is None:
            self.n_data = np.log10(len(self.solutions_df))
        else:
            self.n_data = min(n_data, np.log10(len(self.solutions_df)))
            self.solutions_df = self.solutions_df[:(10**n_data)]
        self.repeats = repeats

    def __len__(self):
        return int(10**self.n_data) * self.repeats

    def __getitem__(self, idx):
        idx = idx % len(self.solutions_df)
        img_name = os.path.join(self.root_dir,
                                self.solutions_df['im_name'].iloc[idx])
        image = io.imread(img_name).astype(np.float32)
        image = np.array(image.transpose([2, 0, 1]))
        solution = self.solutions_df.iloc[idx, 1]
        solution = solution.astype('float32')  # .reshape(-1, 1)
        sample = {'image': image, 'solution': solution}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Function definitions


def loss(x, y):
    x = torch.flatten(x)
    x2 = -torch.logsumexp(torch.stack((x, torch.zeros_like(x))), 0)
    x1 = x+x2
    # x = (torch.exp(x))/(torch.exp(x)+1)
    l1 = -torch.mean(torch.flatten(y)*x1)
    l2 = -torch.mean(torch.flatten(1-y)*x2)
    return l1+l2


def accuracy(x, y):
    x = torch.gt(x, 0.5).float().flatten()
    return torch.mean(torch.eq(x, y.flatten()).float())


def optimize(model, N, lr=0.01, Nkeep=100, momentum=0, clip=np.inf,
             device='cpu'):
    # optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('generating first data')
    x, y = dl.create_training_data(Nkeep)
    x_tensor = torch.tensor(x.transpose((0, 3, 1, 2)), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    print('starting optimization')
    with tqdm.tqdm(total=N, dynamic_ncols=True, smoothing=0.01) as pbar:
        for i in range(N):
            xnew, ynew = dl.create_training_data(1)
            i = np.random.randint(len(x_tensor))
            x_tensor[i] = torch.as_tensor(xnew[0].transpose((2, 0, 1)))
            y_tensor[i] = torch.as_tensor(ynew[0])
            optimizer.zero_grad()
            y_est = model.forward(x_tensor)
            l_batch = loss(y_est, y_tensor)
            l_batch.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            pbar.postfix = '  loss:%0.5f' % l_batch.item()
            pbar.update()
            # if i % 25 == 24:
            #     print(l.item())


def optimize_saved(model, N, root_dir, optimizer,
                   batchsize=20, clip=np.inf, smooth_display=0.9,
                   loss_file=None, tMax=np.inf, smooth_l=0, device='cpu',
                   val_dir=None, check_dir=None, filename=None, n_data=6):
    d = dead_leaves_dataset(root_dir, n_data=n_data, repeats=10**(6-n_data))
    dataload = DataLoader(d, batch_size=batchsize,
                          shuffle=True, num_workers=20)
    print('starting optimization\n')
    if loss_file:
        if os.path.isfile(loss_file):
            losses = np.load(loss_file)
        else:
            losses = np.array([])
    with tqdm.tqdm(total=min(N * len(d), tMax * batchsize * N),
                   dynamic_ncols=True, smoothing=0.01) as pbar:
        k0 = len(losses)
        k = k0
        losses = np.concatenate((losses,
                                 np.zeros(int(N * len(d) / batchsize))))
        for iEpoch in range(N):
            for i, samp in enumerate(dataload):
                k = k+1
                x_tensor = samp['image'].to(device)
                y_tensor = samp['solution'].to(device)
                optimizer.zero_grad()
                y_est = model.forward(x_tensor)
                l_batch = loss(y_est, y_tensor)
                l_batch.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                smooth_l = (smooth_display * smooth_l
                            + (1 - smooth_display) * l_batch.item())
                losses[k-1] = l_batch.item()
                pbar.postfix = ',  loss:%0.5f' % (smooth_l
                                                  / (1-smooth_display**(k-k0)))
                pbar.update(batchsize)
                if loss_file and not (k % 25):
                    np.save(loss_file, losses)
                if k >= tMax:
                    break
            if ((check_dir is not None) and (val_dir is not None)
                    and (filename is not None)):
                save_checkpoint(model, val_dir, filename, check_dir,
                                batchsize=batchsize, device=device)


def overtrain(model, root_dir, optimizer, batchsize=20, clip=np.inf,
              smooth_display=0.9, loss_file=None, tMax=np.inf, smooth_l=0,
              device='cpu'):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d, batch_size=batchsize, shuffle=True, num_workers=6)
    print('starting optimization\n')
    with tqdm.tqdm(total=min(len(d), batchsize * tMax),
                   dynamic_ncols=True, smoothing=0.01) as pbar:
        losses = np.zeros(int(len(d)/batchsize))
        k = 0
        for i, samp in enumerate(dataload):
            k = k + 1
            if i == 0:
                x_tensor = samp['image'].to(device)
                y_tensor = samp['solution'].to(device)
            optimizer.zero_grad()
            y_est = model.forward(x_tensor)
            l_batch = loss(y_est, y_tensor)
            l_batch.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            smooth_l = (smooth_display * smooth_l
                        + (1-smooth_display) * l_batch.item())
            losses[k-1] = l_batch.item()
            pbar.postfix = ',  loss:%0.5f' % (smooth_l/(1-smooth_display**k))
            pbar.update(batchsize)
            if k >= tMax:
                return


def evaluate(model, root_dir, batchsize=20, device='cpu', N_max=np.inf):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d, batch_size=batchsize, shuffle=True, num_workers=2)
    with tqdm.tqdm(total=min(len(d),N_max), dynamic_ncols=True, smoothing=0.01) as pbar:
        with torch.no_grad():
            losses = np.zeros(int(min(len(d),N_max)/batchsize))
            accuracies = np.zeros(int(min(len(d),N_max)/batchsize))
            for i, samp in enumerate(dataload):
                if i >= (N_max / batchsize):
                    break
                x_tensor = samp['image'].to(device)
                y_tensor = samp['solution'].to(device)
                y_est = model.forward(x_tensor)
                l_batch = loss(y_est, y_tensor)
                acc = accuracy(y_est, y_tensor)
                losses[i] = l_batch.item()
                accuracies[i] = acc.item()
                pbar.postfix = ',  loss:%0.5f' % np.mean(losses[:(i+1)])
                pbar.update(batchsize)
    return losses, accuracies


def count_positive(root_dir):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d, batch_size=200, shuffle=True, num_workers=6)
    pos_samples = 0
    all_samples = 0
    for i, samp in enumerate(dataload):
        pos_samples = pos_samples + np.sum(samp['solution'].detach().numpy())
        all_samples = all_samples + len(samp['solution'].detach().numpy())
    return pos_samples, all_samples


def save_checkpoint(model, val_dir, filename, check_dir, batchsize=20,
                    device='cpu'):
    losses, accuracies = evaluate(model,
                                  val_dir,
                                  batchsize=batchsize,
                                  device=device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename + '_' + timestamp
    path = check_dir + filename + '.pt'
    path_l = check_dir + filename + '_l.npy'
    path_acc = check_dir + filename + '_acc.npy'
    torch.save(model.state_dict(), path)
    np.save(path_l, np.array(losses))
    np.save(path_acc, np.array(accuracies))


def plot_loss(check_dir, filename, path_loss, smooth_n=25):
    losses = np.load(path_loss)
    val_loss = []
    val_acc = []
    timestamp = []
    for p in pathlib.Path(check_dir).glob(filename+'_[0-9]*_[0-9]*_l.npy'):
        val_loss.append(np.mean(np.load(p)))
        timestamp.append(int(p.name.split('_')[-3])*1000000
                         + int(p.name.split('_')[-2]))
    order = np.argsort(timestamp)
    val_loss = np.array(val_loss)[order]
    timestamp = []
    check_path = pathlib.Path(check_dir)
    for p in check_path.glob(filename+'_[0-9]*_[0-9]*_acc.npy'):
        val_acc.append(np.mean(np.load(p)))
        timestamp.append(int(p.name.split('_')[-3])*1000000
                         + int(p.name.split('_')[-2]))
    order = np.argsort(timestamp)
    val_acc = np.array(val_acc)[order]
    x_val = np.linspace(len(losses) / len(val_loss),
                        len(losses) - smooth_n,
                        len(val_loss))
    plt.figure()
    plt.plot(np.convolve(losses, np.ones(smooth_n) / smooth_n, 'valid'))
    plt.plot(x_val, val_loss, 'k.-')
    plt.figure()
    plt.plot(val_acc, 'k.-')
    plt.show()


def get_optimal_model(check_dir, model_name, im_size, time, n_neurons,
                      kernel, average, device='cpu'):
    model = get_model(model_name, im_size, time, n_neurons, kernel, average,
                      device)
    val_name = []
    val_loss = []
    val_acc = []
    timestamp = []
    filename = get_filename(model_name, im_size, n_neurons, kernel, time, average)
    for p in pathlib.Path(check_dir).glob(filename+'_[0-9]*_[0-9]*_'+'l.npy'):
        val_name.append(p)
        val_loss.append(np.mean(np.load(p)))
        timestamp.append(int(p.name.split('_')[-3])*1000000
                         + int(p.name.split('_')[-2]))
    if len(val_loss) >= 1:
        idx_min = np.argmin(val_loss)
        val_name = val_name[idx_min]
        model_name = filename.split('_')[2]
        val_name = check_dir + '_'.join(val_name.name.split('_')[:-1])
        val_acc = np.mean(np.load(val_name + '_acc.npy'))
        val_loss = np.mean(np.load(val_name + '_l.npy'))
        model.load_state_dict(torch.load(val_name + '.pt',
                                         map_location=torch.device('cpu')))
    return model, val_acc, val_loss


def calc_results(check_dir='/Users/heiko/deadrects/check_points/',
                 train_dir='/Users/heiko/deadrects/training_%d/',
                 val_dir='/Users/heiko/deadrects/validation_%d/',
                 test_dir='/Users/heiko/deadrects/test_%d/',
                 model_name='B', time=5, n_neurons=10, kernel=3,
                 average=False, device='cpu'):
    results = np.zeros((3, 5, 2))
    k = 0
    for im_size in [3, 5, 10, 30, 100]:
        print("started model '%s', imsize=%d\n" % (model_name, im_size))
        train_dir_i = train_dir % im_size
        val_dir_i = val_dir % im_size
        test_dir_i = test_dir % im_size
        model = get_optimal_model(check_dir, model_name, im_size, time,
                                  n_neurons, kernel, average, device)[0]
        loss_train, acc_train = evaluate(model, train_dir_i,
                                         batchsize=100, N_max=10000,
                                         device=device)
        results[0, k, 0] = np.mean(loss_train)
        results[0, k, 1] = np.mean(acc_train)
        loss_val, acc_val = evaluate(model, val_dir_i,
                                     batchsize=100, N_max=10000,
                                     device=device)
        results[1, k, 0] = np.mean(loss_val)
        results[1, k, 1] = np.mean(acc_val)
        loss_test, acc_test = evaluate(model, test_dir_i,
                                       batchsize=100, N_max=10000,
                                       device=device)
        results[2, k, 0] = np.mean(loss_test)
        results[2, k, 1] = np.mean(acc_test)
        k += 1
    return results


def save_results(check_dir='/Users/heiko/deadrects/check_points/',
                 train_dir='/Users/heiko/deadrects/training_%d/',
                 val_dir='/Users/heiko/deadrects/validation_%d/',
                 test_dir='/Users/heiko/deadrects/test_%d/',
                 time=5, n_neurons=10, kernel=3,
                 average=False, device='cpu'):
    if average:
        res_file_name = ('/Users/heiko/deadrects/results_t%d_nn%02d_k%d_avg.npy'
                         % (time, n_neurons, kernel))
    else:
        res_file_name = ('/Users/heiko/deadrects/results_t%d_nn%02d_k%d.npy'
                         % (time, n_neurons, kernel))
    results = np.zeros((4, 3, 5, 2))
    results[0] = calc_results(check_dir=check_dir, train_dir=train_dir,
                              val_dir=val_dir, test_dir=test_dir, time=time,
                              n_neurons=n_neurons, kernel=kernel,
                              average=average, device=device,
                              model_name='B')
    results[1] = calc_results(check_dir=check_dir, train_dir=train_dir,
                              val_dir=val_dir, test_dir=test_dir, time=time,
                              n_neurons=n_neurons, kernel=kernel,
                              average=average, device=device,
                              model_name='BL')
    results[2] = calc_results(check_dir=check_dir, train_dir=train_dir,
                              val_dir=val_dir, test_dir=test_dir, time=time,
                              n_neurons=n_neurons, kernel=kernel,
                              average=average, device=device,
                              model_name='BT')
    results[3] = calc_results(check_dir=check_dir, train_dir=train_dir,
                              val_dir=val_dir, test_dir=test_dir, time=time,
                              n_neurons=n_neurons, kernel=kernel,
                              average=average, device=device,
                              model_name='BLT')
    np.save(res_file_name, results)


def plot_results(time=5, n_neurons=10, kernel=3, average=False):
    if average:
        res_file_name = ('/Users/heiko/deadrects/results_t%d_nn%02d_k%d_avg.npy'
                         % (time, n_neurons, kernel))
    else:
        res_file_name = ('/Users/heiko/deadrects/results_t%d_nn%02d_k%d.npy'
                         % (time, n_neurons, kernel))
    results = np.load(res_file_name)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(results[i, :, :, 0].T)
        plt.ylabel('Loss')
        plt.xticks(range(5), [3, 5, 10, 30, 100])
        plt.xlabel('Image size [px]')
        if i == 0:
            plt.title('B')
        elif i == 1:
            plt.title('BL')
        elif i == 2:
            plt.title('BT')
        elif i == 3:
            plt.title('BLT')
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(results[i, :, :, 1].T)
        plt.ylabel('Accuracy')
        plt.xticks(range(5), [3, 5, 10, 30, 100])
        plt.xlabel('Image size [px]')
        plt.ylim([.5, 1])
        if i == 0:
            plt.title('B')
        elif i == 1:
            plt.title('BL')
        elif i == 2:
            plt.title('BT')
        elif i == 3:
            plt.title('BLT')
    fig = plt.figure(figsize=[9, 6])
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(results[:, 2, :, 1].T, 's-', linewidth=2)
    res_human = load_validation_results()
    plt.plot(res_human[:, 0] / res_human[:, 1], 'ks-', linewidth=2)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xticks(range(5), [3, 5, 10, 30, 100])
    plt.xlabel('Image size [px]', fontsize=18)
    plt.ylim([.5, 1])
    if average:
        plt.title('Average: Nn=%d' % (n_neurons), fontsize=24)
    else:
        plt.title('Linear: Nn=%d' % (n_neurons))
    leg = plt.legend(['B', 'BL', 'BT', 'BLT', 'Human'],
                     frameon=False, loc='lower center',
                     fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig('Figures/BLT_comparison.pdf')


def load_validation_results():
    k = 0
    results = np.zeros((5, 2))
    for im_size in [3, 5, 10, 30, 100]:
        filename = 'result_%d_' % im_size
        res_list = []
        for p in pathlib.Path('Experiment/resultsValidation/Heiko/').glob(filename+'*.npy'):
            res = np.load(p)
            res_list.append(res)
        res_all = np.concatenate(res_list, axis=0)
        results[k] = [np.sum(res_all[:, 2] == res_all[:, 3]), res_all.shape[0]]
        k += 1
    return results


def get_model(model_name, im_size, time, n_neurons, kernel, average, device):
    if model_name == 'model':
        model = model_class(im_size).to(device)
    elif model_name == 'model2':
        model = model2_class(im_size).to(device)
    elif model_name == 'pred':
        model = model_pred_like().to(device)
    elif model_name == 'recurrent':
        model = model_recurrent(n_rep=time).to(device)
    elif model_name == 'min':
        model = model_min_class().to(device)
    elif model_name == 'res':
        model = model_residual().to(device)
    elif model_name == 'BLT':
        model = model_BLT(im_size, n_rep=time, n_neurons=n_neurons,
                          kernel=kernel, L=True, T=True,
                          average=average).to(device)
    elif model_name == 'BL':
        model = model_BLT(im_size, n_rep=time, n_neurons=n_neurons,
                          kernel=kernel, L=True, T=False,
                          average=average).to(device)
    elif model_name == 'BT':
        model = model_BLT(im_size, n_rep=time, n_neurons=n_neurons,
                          kernel=kernel, L=False, T=True,
                          average=average).to(device)
    elif model_name == 'B':
        model = model_BLT(im_size, n_rep=time, n_neurons=n_neurons,
                          kernel=kernel, L=False, T=False,
                          average=average).to(device)
    return model


def get_filename(model_name, im_size, n_neurons, kernel, time, average, n_data):
    filename = 'model_%d_%s' % (im_size, model_name)
    if not n_neurons == 10:
        filename = filename + '_nn%02d' % n_neurons
    if not kernel == 3:
        filename = filename + '_k%02d' % kernel
    if not time == 5:
        filename = filename + '_%02d' % time
    if not n_data == 6:
        filename = filename + '_d%02d' % n_data
    if average:
        filename = filename + '_avg'
    return filename


def main(model_name, action, average_neighbors=False,
         device='cpu', weight_decay=10**-3, epochs=1, lr=0.001,
         tMax=np.inf, batchsize=20, time=5, n_neurons=10,
         kernel=3, im_size=5, average=False, n_data=6):
    model = get_model(model_name, im_size, time, n_neurons,
                      kernel, average, device)
    filename = get_filename(model_name, im_size, n_neurons, kernel, time,
                            average, n_data)
    check_dir = '/Users/heiko/deadrects/check_points/'
    data_folder = '/Users/heiko/deadrects/training_%d/' % im_size
    val_dir = '/Users/heiko/deadrects/validation_%d/' % im_size
    path = '/Users/heiko/deadrects/models/' + filename + '.pt'
    path_opt = '/Users/heiko/deadrects/models/' + filename + '_opt.pt'
    path_loss = '/Users/heiko/deadrects/models/' + filename + '_loss.npy'
    path_l = '/Users/heiko/deadrects/models/' + filename + '_l.npy'
    path_acc = '/Users/heiko/deadrects/models/' + filename + '_acc.npy'
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 amsgrad=True,
                                 weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),lr = lr,weight_decay=weight_decay)
    if action == 'reset':
        os.remove(path)
        os.remove(path_opt)
        os.remove(path_loss)
        os.remove(path_l)
        os.remove(path_acc)
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        optimizer.load_state_dict(torch.load(path_opt, map_location=torch.device('cpu')))
        optimizer.param_groups[0]['lr'] = lr
    if action == 'train':
        optimize_saved(model, epochs, data_folder, optimizer,
                       batchsize=batchsize, clip=np.inf, smooth_display=0.9,
                       loss_file=path_loss, tMax=tMax, device=device,
                       check_dir=check_dir, val_dir=val_dir, filename=filename,
                       n_data=n_data)
        torch.save(model.state_dict(), path)
        torch.save(optimizer.state_dict(), path_opt)
        return model
    elif action == 'eval':
        l, acc = evaluate(model, val_dir, batchsize=batchsize, device=device)
        np.save(path_l, np.array(l))
        np.save(path_acc, np.array(acc))
        return acc, l
    elif action == 'overtrain':
        overtrain(model, data_folder, optimizer,
                  batchsize=batchsize, clip=np.inf, smooth_display=0.9,
                  loss_file=None, tMax=tMax, device=device)
    elif action == 'print':
        print(filename)
        if os.path.isfile(path_l):
            print('negative log-likelihood:')
            print('%.6f' % np.mean(np.load(path_l)))
            print('accuracy:')
            print('%.4f %%' % (100*np.mean(np.load(path_acc))))
        else:
            print('not yet evaluated!')
    elif action == 'plot_loss':
        plot_loss(check_dir, filename, path_loss, smooth_n=batchsize)


# Testing the evaluation for imagenet
# mTest = minimumNet().to(device)
# mEvalTest,accTest,lTest = evaluate_acc(mTest,data_loader,val_loader,
#                                        device=device,maxiter=np.inf)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device",
                        help="device to run on [cuda,cpu]",
                        choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument("-E", "--epochs",
                        help="numer of epochs",
                        type=int, default=1)
    parser.add_argument("-b", "--batch_size",
                        help="size of a batch",
                        type=int, default=20)
    parser.add_argument("-w", "--weight_decay",
                        help="how much weight decay?",
                        type=float, default=0)
    parser.add_argument("-l", "--learning_rate",
                        help="learning rate",
                        type=float, default=10**-3)
    parser.add_argument("-s", "--tMax",
                        help="maximum number of training steps",
                        type=int, default=np.inf)
    parser.add_argument("-t", "--time",
                        help="number of timesteps",
                        type=int, default=5)
    parser.add_argument("-n", "--n_neurons",
                        help="number of neurons/features",
                        type=int, default=10)
    parser.add_argument("-i", "--im_size",
                        help="image size",
                        type=int, default=5)
    parser.add_argument("-k", "--kernel",
                        help="kernel size",
                        type=int, default=3)
    parser.add_argument("-a", "--n_data",
                        help="log10 number of samples used for training [1-6]",
                        type=int, default=6)
    parser.add_argument('--average', dest='average', action='store_true')
    parser.add_argument("action",
                        help="what to do? [train, eval, overtrain, print"
                             + ", reset, plot_loss]",
                        choices=['train', 'eval', 'overtrain', 'print',
                                 'reset', 'plot_loss'])
    parser.add_argument("model_name",
                        help="model to be trained [model, deep, recurrent,"
                             + " pred, res, min, model2, BLT]",
                        choices=['model', 'model2', 'deep', 'res', 'recurrent',
                                 'pred', 'min', 'BLT', 'BT', 'BL', 'B'])
    parser.set_defaults(average=False)
    args = parser.parse_args()
    main(args.model_name, args.action, device=args.device,
         weight_decay=float(args.weight_decay), epochs=args.epochs,
         lr=args.learning_rate, tMax=args.tMax, batchsize=args.batch_size,
         time=args.time, n_neurons=args.n_neurons, kernel=args.kernel,
         im_size=args.im_size, average=args.average, n_data=args.n_data)
