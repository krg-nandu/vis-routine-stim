#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:45:25 2019

@author: heiko
Neural network training pytorch variant
"""
import sys, getopt
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
from skimage import io #, transform
from torch.utils.data import Dataset, DataLoader

eps = 10**-5

## standard parameters

sizes = sizes=np.arange(1,6,dtype=np.float)
imSize = np.array([5,5])


## Model definitions

def init_weights_layer_conv(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

def init_weights_layer_linear(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.1*nn.init.calculate_gain('relu'))
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
    def __init__(self):
        super(model_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(3*5*5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 3 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)

class model_residual(nn.Module):
    def __init__(self):
        super(model_residual, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5,padding=(2,2))
        self.conv1_1 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv1_2 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv1_3 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv1_4 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv1_5 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv1_6 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3,padding=(1,1))
        self.conv2_1 = nn.Conv2d(40, 40, kernel_size=3,padding=(1,1))
        self.conv2_2 = nn.Conv2d(40, 40, kernel_size=3,padding=(1,1))
        self.conv2_3 = nn.Conv2d(40, 40, kernel_size=3,padding=(1,1))
        self.conv2_4 = nn.Conv2d(40, 40, kernel_size=3,padding=(1,1))
        self.conv2_5 = nn.Conv2d(40, 40, kernel_size=3,padding=(1,1))
        self.conv2_6 = nn.Conv2d(40, 40, kernel_size=3,padding=(1,1))
        self.fc1 = nn.Linear(160, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
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
        x = x2_2.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)

class model2_class(nn.Module):
    def __init__(self):
        super(model2_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(3*5*5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)
    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 3 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)

class model_deep_class(nn.Module):
    def __init__(self):
        super(model_deep_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(3*5*5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc9 = nn.Linear(100, 1)
    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 3 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_linear)


class model_BLT(nn.Module):
    def __init__(self,n_rep=5,n_neurons=10,kernel=3,L=True,T=True):
        super(model_BLT, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        # Bottom-up
        self.conv0_1 = nn.Conv2d(3, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)))
        self.conv1_2 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)))
        self.conv2_3 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)))
        if L:
            # Lateral
            self.conv1_1 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)),bias=False)
            self.conv2_2 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)),bias=False)
            self.conv3_3 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)),bias=False)
        if T:
            # top-down
            self.conv2_1 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)),bias=False)
            self.conv3_2 = nn.Conv2d(n_neurons, n_neurons, kernel_size=kernel, padding=(int((kernel-1)/2),int((kernel-1)/2)),bias=False)
        self.fc1 = nn.Linear(n_neurons*5*5, 1)
        self.n_rep = n_rep
        self.n_neurons = n_neurons
        self.L = L
        self.T = T
    def forward(self, x):
        x = self.norm(x)
        h1 = torch.zeros([x.shape[0],self.n_neurons,5,5],device=x.device)
        h2 = torch.zeros([x.shape[0],self.n_neurons,5,5],device=x.device)
        h3 = torch.zeros([x.shape[0],self.n_neurons,5,5],device=x.device)
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
        x = h3.view(-1,self.n_neurons*5*5)
        x = self.fc1(x)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_recurrent(nn.Module):
    def __init__(self,n_rep=20,n_neurons=10000):
        super(model_recurrent, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc1 = nn.Linear(n_neurons+(3*5*5), n_neurons)
        self.fc2 = nn.Linear(n_neurons, 1)
        self.n_rep = n_rep
        self.n_neurons = n_neurons
    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1,3*5*5)
        siz = list(x.shape)
        siz[1] = self.n_neurons
        h1 = torch.ones(siz,device=x.device)
        for i in range(self.n_rep):
            inp1 = torch.cat((x,(h1-torch.mean(h1,dim=1).view(-1,1))/(eps+h1.std())),dim=1)
            h1 = F.relu(self.fc1(inp1))
        x = self.fc2(h1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


def get_shifted_values(feat,neighbors):
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


class model_pred_like(nn.Module):
    def __init__(self,n_rep=5,neighbors=np.array([[0,-1],[0,1],[1,0],[-1,0]])):
        super(model_pred_like, self).__init__()
        self.neighbors = neighbors
        self.norm = nn.InstanceNorm2d(3)
        self.conv1value = nn.Conv2d(3, 10, kernel_size=5,stride=1,padding=(2,2))
        self.conv1prec = nn.Conv2d(3, 10, kernel_size=5,stride=1,padding=(2,2))
        self.conv1neigh = nn.Conv2d(3, 4, kernel_size=5,stride=1,padding=(2,2))
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2value = nn.Conv2d(10, 10, kernel_size=3,stride=1,padding=(1,1))
        self.conv2prec = nn.Conv2d(10, 10, kernel_size=3,stride=1,padding=(1,1))
        self.conv2neigh = nn.Conv2d(10, 4, kernel_size=3,stride=1,padding=(1,1))
        self.pool2 = nn.MaxPool2d(6, 6)
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
        self.logC1 = torch.nn.Parameter(torch.Tensor(-2*np.ones((self.neighbors.shape[0],10))))
        self.register_parameter('logC_1', self.logC1)
        self.logC2 = torch.nn.Parameter(torch.Tensor(-2*np.ones((self.neighbors.shape[0],10))))
        self.register_parameter('logC_2', self.logC2)
        self.n_rep = n_rep
    def forward(self, x):
        x = self.norm(x)
        epsilon = 0.000001
        value1in = F.relu(self.conv1value(x))
        prec1in = F.relu(self.conv1prec(x))+ epsilon
        neigh1 = F.relu(self.conv1neigh(x)).unsqueeze(0).permute((2,1,0,3,4))
        value1 = value1in
        prec1 = prec1in 
        out1 = self.pool(value1)
        value2in = F.relu(self.conv2value(out1))
        prec2in = F.relu(self.conv2prec(out1))+ epsilon
        neigh2 = F.relu(self.conv2neigh(out1)).unsqueeze(0).permute((2,1,0,3,4))
        value2 = value2in
        prec2 = prec2in 
        neighbors = self.neighbors
        C1 = torch.exp(self.logC1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        C2 = torch.exp(self.logC2).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        for i in range(self.n_rep):
            neighValues1 = get_shifted_values(value1,neighbors)
            neighPrec1 = get_shifted_values(prec1,neighbors)
            neighPrec1 = (neigh1*C1*neighPrec1)/(neigh1*C1 + neighPrec1+epsilon)
            prec1 = prec1in + torch.sum(neighPrec1,0)
            value1 = (prec1in*value1in + torch.sum(neighPrec1*neighValues1,0))/prec1
            out1 = self.pool(value1)
            value2in = F.relu(self.conv2value(out1))
            prec2in = F.relu(self.conv2prec(out1))+ epsilon
            neigh2 = F.relu(self.conv2neigh(out1)).unsqueeze(0).permute((2,1,0,3,4))
            neighValues2 = get_shifted_values(value2,neighbors)
            neighPrec2 = get_shifted_values(prec2,neighbors)
            neighPrec2 = (neigh2*C2*neighPrec2)/(neigh2*C2 + neighPrec2+epsilon)
            prec2 = prec2in + torch.sum(neighPrec2,0)
            value2 = (prec2in*value2in + torch.sum(neighPrec2*neighValues2,0))/prec2
            out2 = self.pool(value2)
        x = out2.view(-1, 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


## Dataset definition (for reading from disk)
class dead_leaves_dataset(Dataset):
    """dead leaves dataset."""
    def __init__(self, root_dir, transformation=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.solutions_df = pd.read_csv(os.path.join(root_dir,'solution.csv'),index_col=0)
        self.root_dir = root_dir
        self.transform = transformation

    def __len__(self):
        return len(self.solutions_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.solutions_df['im_name'].iloc[idx])
        image = io.imread(img_name).astype(np.float32)
        image = np.array(image.transpose([2,0,1]))
        solution = self.solutions_df.iloc[idx, 1]
        solution = solution.astype('float32') #.reshape(-1, 1)
        sample = {'image': image, 'solution': solution}

        if self.transform:
            sample = self.transform(sample)

        return sample

## Function definitions

def loss(x,y):
    x = torch.flatten(x)
    x2 = -torch.logsumexp(torch.stack((x,torch.zeros_like(x))),0)
    x1 = x+x2
    #x = (torch.exp(x))/(torch.exp(x)+1)
    l1 = -torch.mean(torch.flatten(y)*x1)
    l2 = -torch.mean(torch.flatten(1-y)*x2)
    return l1+l2


def accuracy(x,y):
    x = torch.gt(x,0.5).float().flatten()
    return torch.mean(torch.eq(x,y.flatten()).float())


def optimize(model,N,lr=0.01,Nkeep=100,momentum=0,clip=np.inf, device='cpu'):
    # optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('generating first data')
    x,y = dl.create_training_data(Nkeep)
    x_tensor = torch.tensor(x.transpose((0,3,1,2)),dtype=torch.float32)
    y_tensor = torch.tensor(y,dtype=torch.float32)
    print('starting optimization')
    with tqdm.tqdm(total=N, dynamic_ncols=True,smoothing=0.01) as pbar:
      for i in range(N):
        xnew,ynew = dl.create_training_data(1)
        i = np.random.randint(len(x_tensor))
        x_tensor[i]=torch.as_tensor(xnew[0].transpose((2,0,1)))
        y_tensor[i]=torch.as_tensor(ynew[0])
        optimizer.zero_grad()
        y_est = model.forward(x_tensor)
        l = loss(y_est,y_tensor)
        l.backward()
        nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        pbar.postfix = '  loss:%0.5f' % l.item()
        pbar.update()
        #if i % 25 == 24:
        #    print(l.item())


def optimize_saved(model, N, root_dir, optimizer,
                   batchsize=20, clip=np.inf, smooth_display=0.9,
                   loss_file=None, kMax=np.inf, smooth_l = 0, device='cpu',
                   val_dir=None, check_dir=None, filename=None):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=6)
    print('starting optimization\n')
    if loss_file:
        if os.path.isfile(loss_file):
            losses = np.load(loss_file)
        else:
            losses = np.array([])
    with tqdm.tqdm(total=min(N*len(d),kMax*batchsize*N), dynamic_ncols=True,smoothing=0.01) as pbar:
        k0 = len(losses)
        k = k0
        losses = np.concatenate((losses,np.zeros(int(N*len(d)/batchsize))))
        for iEpoch in range(N):
            for i,samp in enumerate(dataload):
                k=k+1
                x_tensor = samp['image'].to(device)
                y_tensor = samp['solution'].to(device)
                optimizer.zero_grad()
                y_est = model.forward(x_tensor)
                l = loss(y_est,y_tensor)
                l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                smooth_l = smooth_display*smooth_l+(1-smooth_display) * l.item()
                losses[k-1]=l.item()
                pbar.postfix = ',  loss:%0.5f' % (smooth_l/(1-smooth_display**(k-k0)))
                pbar.update(batchsize)
                if loss_file and not (k%25):
                    np.save(loss_file,losses)
                if k>=kMax:
                    break
            if (check_dir is not None) and (val_dir is not None) and (filename is not None):
                save_checkpoint(model, val_dir, filename, check_dir, batchsize=batchsize, device=device)


def overtrain(model,root_dir,optimizer,batchsize=20,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=np.inf,smooth_l = 0, device='cpu'):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=6)
    print('starting optimization\n')
    with tqdm.tqdm(total=min(len(d),batchsize*kMax), dynamic_ncols=True,smoothing=0.01) as pbar:
        losses = np.zeros(int(len(d)/batchsize))
        k = 0
        for i,samp in enumerate(dataload):
            k=k+1
            if i == 0:
                x_tensor = samp['image'].to(device)
                y_tensor = samp['solution'].to(device)
            optimizer.zero_grad()
            y_est = model.forward(x_tensor)
            l = loss(y_est,y_tensor)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            smooth_l = smooth_display*smooth_l+(1-smooth_display) * l.item()
            losses[k-1]=l.item()
            pbar.postfix = ',  loss:%0.5f' % (smooth_l/(1-smooth_display**k))
            pbar.update(batchsize)
            if k>=kMax:
                return


def evaluate(model,root_dir,batchsize=20, device='cpu'):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=2)
    with tqdm.tqdm(total=len(d), dynamic_ncols=True,smoothing=0.01) as pbar:
        with torch.no_grad():
            losses = np.zeros(int(len(d)/batchsize))
            accuracies = np.zeros(int(len(d)/batchsize))
            for i,samp in enumerate(dataload):
                x_tensor = samp['image'].to(device)
                y_tensor = samp['solution'].to(device)
                y_est = model.forward(x_tensor)
                l = loss(y_est,y_tensor)
                acc = accuracy(y_est,y_tensor)
                losses[i]=l.item()
                accuracies[i] = acc.item()
                pbar.postfix = ',  loss:%0.5f' % np.mean(losses[:(i+1)])
                pbar.update(batchsize)
    return losses, accuracies


def count_positive(root_dir):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=200,shuffle=True,num_workers=6)
    pos_samples = 0
    all_samples = 0
    for i,samp in enumerate(dataload):
        pos_samples = pos_samples + np.sum(samp['solution'].detach().numpy())
        all_samples = all_samples + len(samp['solution'].detach().numpy())
    return pos_samples,all_samples


def save_checkpoint(model, val_dir, filename, check_dir, batchsize=20, device='cpu'):
    losses, accuracies = evaluate(model,
                                  val_dir,
                                  batchsize=batchsize,
                                  device=device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename + '_' + timestamp
    path= check_dir + filename + '.pt'
    path_l= check_dir + filename + '_l.npy'
    path_acc= check_dir + filename + '_acc.npy'
    torch.save(model.state_dict(),path)
    np.save(path_l,np.array(losses))
    np.save(path_acc,np.array(accuracies))


def plot_loss(check_dir, filename, path_loss, smooth_n=25):
    losses = np.load(path_loss)
    val_loss = []
    val_acc = []
    timestamp = []
    for p in pathlib.Path(check_dir).glob(filename+'_*'+'l.npy'):
        val_loss.append(np.mean(np.load(p)))
        timestamp.append(int(p.name.split('_')[-3])*1000000+int(p.name.split('_')[-2]))
    order = np.argsort(timestamp)
    val_loss = np.array(val_loss)[order]
    timestamp = []
    for p in pathlib.Path(check_dir).glob(filename+'_*'+'acc.npy'):
        val_acc.append(np.mean(np.load(p)))
        timestamp.append(int(p.name.split('_')[-3])*1000000+int(p.name.split('_')[-2]))
    order = np.argsort(timestamp)
    val_acc = np.array(val_acc)[order]
    x_val = np.linspace(len(losses)/len(val_loss),len(losses)-smooth_n,len(val_loss))
    plt.figure()
    plt.plot(np.convolve(losses,np.ones(smooth_n)/smooth_n,'valid'))
    plt.plot(x_val,val_loss,'k.-')
    plt.figure()
    plt.plot(val_acc,'k.-')
    plt.show()


def main(model_name,action,average_neighbors=False,device='cpu',weight_decay = 10**-3,epochs=1,lr = 0.001,kMax=np.inf,batchsize=20,time=5,n_neurons=10,kernel=3):
    filename = 'model_tiny_%s' % model_name
    if model_name == 'model':
        model = model_class().to(device)
    elif model_name == 'deep':
        model = model_deep_class().to(device)
    elif model_name == 'model2':
        model = model2_class().to(device)
    elif model_name == 'pred':
        model = model_pred_like().to(device)
    elif model_name == 'recurrent':
        model = model_recurrent(n_rep=time).to(device)
    elif model_name == 'min':
        model = model_min_class().to(device)
    elif model_name == 'res':
        model = model_residual().to(device)
    elif model_name == 'BLT':
        model = model_BLT(n_rep=time,n_neurons=n_neurons,kernel=kernel,L=True,T=True).to(device)
    elif model_name == 'BL':
        model = model_BLT(n_rep=time,n_neurons=n_neurons,kernel=kernel,L=True,T=False).to(device)
    elif model_name == 'BT':
        model = model_BLT(n_rep=time,n_neurons=n_neurons,kernel=kernel,L=False,T=True).to(device)
    elif model_name == 'B':
        model = model_BLT(n_rep=time,n_neurons=n_neurons,kernel=kernel,L=False,T=False).to(device)
    if not n_neurons==10:
        filename = filename + '_nn%02d' % n_neurons
    if not kernel==3:
        filename = filename + '_k%02d' % kernel
    if not time==5:
        filename = filename + '_%02d' % time
    check_dir = '/Users/heiko/tinytinydeadrects/check_points/'
    val_dir = '/Users/heiko/tinytinydeadrects/validation'
    path= '/Users/heiko/tinytinydeadrects/models/' + filename + '.pt'
    path_opt= '/Users/heiko/tinytinydeadrects/models/' + filename + '_opt.pt'
    path_loss= '/Users/heiko/tinytinydeadrects/models/' + filename + '_loss.npy'
    path_l= '/Users/heiko/tinytinydeadrects/models/' + filename + '_l.npy'
    path_acc= '/Users/heiko/tinytinydeadrects/models/' + filename + '_acc.npy'
    optimizer = torch.optim.Adam(model.parameters(),lr = lr,amsgrad=True,weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(),lr = lr,weight_decay=weight_decay)
    if action == 'reset':
        os.remove(path)
        os.remove(path_opt)
        os.remove(path_loss)
        os.remove(path_l)
        os.remove(path_acc)
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        optimizer.load_state_dict(torch.load(path_opt))
        optimizer.param_groups[0]['lr'] = lr
    if action == 'train':
        data_folder = '/Users/heiko/tinytinydeadrects/training'
        optimize_saved(model,epochs,data_folder,optimizer,batchsize=batchsize,
                       clip=np.inf,smooth_display=0.9,loss_file=path_loss,
                       kMax=kMax,device=device,
                       check_dir=check_dir, val_dir=val_dir, filename=filename)
        torch.save(model.state_dict(),path)
        torch.save(optimizer.state_dict(),path_opt)
        return model
    elif action =='eval':
        data_folder = '/Users/heiko/tinytinydeadrects/validation'
        l,acc = evaluate(model,data_folder,batchsize=batchsize,device=device)
        np.save(path_l,np.array(l))
        np.save(path_acc,np.array(acc))
        return acc,l
    elif action =='overtrain':
        data_folder = '/Users/heiko/tinytinydeadrects/training'
        overtrain(model,data_folder,optimizer,batchsize=batchsize,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=kMax,device=device)
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


### Testing the evaluation for imagenet
#mTest = minimumNet().to(device)
#mEvalTest,accTest,lTest = evaluate_acc(mTest,data_loader,val_loader,device=device,maxiter=np.inf)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--device", help="device to run on [cuda,cpu]", choices=['cuda', 'cpu'],default='cpu')
    parser.add_argument("-E","--epochs", help="numer of epochs", type = int ,default=1)
    parser.add_argument("-b","--batch_size", help="size of a batch", type = int ,default=20)
    parser.add_argument("-w","--weight_decay",type=float,help="how much weight decay?",default=0)
    parser.add_argument("-l","--learning_rate",type=float,help="learning rate",default=10**-3)
    parser.add_argument("-k","--kMax",type=int,help="maximum number of training steps",default=np.inf)
    parser.add_argument("-t","--time",type=int,help="number of timesteps",default=5)
    parser.add_argument("-n","--n_neurons",type=int,help="number of neurons/features",default=10)
    parser.add_argument("--kernel",type=int,help="kernel size",default=3)
    parser.add_argument("action",help="what to do? [train,eval,overtrain,print,reset,plot_loss]", choices=['train', 'eval', 'overtrain', 'print', 'reset', 'plot_loss'])
    parser.add_argument("model_name",help="model to be trained [model,deep,recurrent,pred,res,min,model2,BLT]", choices=['model','model2', 'deep','res','recurrent','pred','min','BLT','BT','BL','B'])
    args=parser.parse_args()
    main(args.model_name,args.action,device=args.device,weight_decay=float(args.weight_decay),epochs=args.epochs,
         lr = args.learning_rate,kMax=args.kMax,batchsize=args.batch_size,time=args.time,n_neurons=args.n_neurons,kernel=args.kernel)
