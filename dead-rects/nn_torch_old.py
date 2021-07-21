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

import tqdm

import DeadLeaf as dl

# for data loading
import pandas as pd
from skimage import io #, transform
from torch.utils.data import Dataset, DataLoader

## standard parameters

sizes = 5*np.arange(1,80,dtype='float')
imSize = np.array([300,300])
distances = [5,10,20,40,80]
distancesd = [4,7,14,28,57]


## Model definitions

def init_weights_layer_conv(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

def init_weights_layer_linear(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)
        
class model_class(nn.Module):
    def __init__(self):
        super(model_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5,stride=2)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*3*3, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.exp(x)/(torch.exp(x)+1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)
       
class model2_class(nn.Module):
    def __init__(self):
        super(model2_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5,stride=2)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(20, 100, 5)
        self.conv3 = nn.Conv2d(100, 200, 5)
        self.fc1 = nn.Linear(200*3*3, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 200 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.exp(x)/(torch.exp(x)+1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)

class model_deep_class(nn.Module):
    def __init__(self):
        super(model_deep_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7,stride=2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=7,stride=2)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(40, 100, 5,stride=2)
        self.conv4 = nn.Conv2d(100, 200, 5,stride=2)
        self.conv5 = nn.Conv2d(200, 200, 5,stride=2)
        self.fc1 = nn.Linear(200*6*6, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 200 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.exp(x)/(torch.exp(x)+1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_recurrent(nn.Module):
    def __init__(self,Nrep=5):
        super(model_recurrent, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.norm1 = nn.InstanceNorm2d(20)
        self.norm2 = nn.InstanceNorm2d(40)
        self.conv1 = nn.Conv2d(23, 20, kernel_size=7,stride=1,padding=(3,3))
        self.conv2 = nn.Conv2d(60, 40, kernel_size=7,stride=1,padding=(3,3))
        self.pool = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(40*12*12, 200)
        self.fc2 = nn.Linear(200, 1)
        self.Nrep = Nrep
    def forward(self, x):
        x = self.norm(x)
        siz = list(x.shape)
        siz[1] = 20
        h1 = torch.zeros(siz)
        siz[1] = 40
        siz[2] = int(siz[2]/5)
        siz[3] = int(siz[3]/5)
        h2 = torch.zeros(siz)
        for i in range(self.Nrep):
            inp1 = torch.cat((x,h1),dim=1)
            h1 = F.relu(self.conv1(inp1))
            inp2 = torch.cat((self.pool(h1),h2),dim=1)
            h2 = F.relu(self.conv2(inp2))
        o2 = self.pool(h2)
        x = o2.view(-1, 40 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.exp(x)/(torch.exp(x)+1)
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
    def __init__(self,Nrep=5,neighbors=np.array([[0,-1],[0,1],[1,0],[-1,0]])):
        super(model_pred_like, self).__init__()
        self.neighbors = neighbors
        self.norm = nn.InstanceNorm2d(3)
        self.conv1value = nn.Conv2d(3, 10, kernel_size=5,stride=1,padding=(2,2))
        self.conv1prec = nn.Conv2d(3, 10, kernel_size=5,stride=1,padding=(2,2))
        self.conv1neigh = nn.Conv2d(3, 4, kernel_size=5,stride=1,padding=(2,2))
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2value = nn.Conv2d(10, 10, kernel_size=5,stride=1,padding=(2,2))
        self.conv2prec = nn.Conv2d(10, 10, kernel_size=5,stride=1,padding=(2,2))
        self.conv2neigh = nn.Conv2d(10, 4, kernel_size=5,stride=1,padding=(2,2))
        self.fc1 = nn.Linear(10*12*12, 10)
        self.fc2 = nn.Linear(10, 1)
        
        self.logC1 = torch.nn.Parameter(torch.Tensor(-2*np.ones((self.neighbors.shape[0],10))))
        self.register_parameter('logC_1', self.logC1)
        self.logC2 = torch.nn.Parameter(torch.Tensor(-2*np.ones((self.neighbors.shape[0],10))))
        self.register_parameter('logC_2', self.logC2)
        
        self.Nrep = Nrep
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
        for i in range(self.Nrep):
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
        x = out2.view(-1, 10 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.exp(x)/(torch.exp(x)+1)
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
    l1 = -torch.mean(torch.flatten(y)*torch.log(torch.flatten(x)))
    l2 = -torch.mean(torch.flatten(1-y)*torch.log(1-torch.flatten(x)))
    return l1+l2

def accuracy(x,y):
    x = torch.gt(x,0.5).float().flatten()
    return torch.mean(torch.eq(x,y.flatten()).float())

def optimize(model,N,lr=0.01,Nkeep=100,momentum=0,clip=np.inf):
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

def optimize_saved(model,N,root_dir,lr=0.01,momentum=0,batchsize=20,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=np.inf,smooth_l = 0):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=6)
    # optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    print('starting optimization\n')
    with tqdm.tqdm(total=min(N*len(d),kMax*batchsize), dynamic_ncols=True,smoothing=0.01) as pbar:
        losses = np.zeros(int(N*len(d)/batchsize))
        k = 0
        for iEpoch in range(N):
            for i,samp in enumerate(dataload):
                k=k+1
                x_tensor = samp['image']
                y_tensor = samp['solution']
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
                if loss_file and not (k%25):
                    np.save(loss_file,losses)
                if k>=kMax:
                    return

def overtrain(model,root_dir,lr=0.01,momentum=0,batchsize=20,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=np.inf,smooth_l = 0):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=6)
    # optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('starting optimization\n')
    with tqdm.tqdm(total=min(len(d),batchsize*kMax), dynamic_ncols=True,smoothing=0.01) as pbar:
        losses = np.zeros(int(len(d)/batchsize))
        k = 0
        for i,samp in enumerate(dataload):
            k=k+1
            if i == 0:
                x_tensor = samp['image']
                y_tensor = samp['solution']
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
            if loss_file and not (k%25):
                np.save(loss_file,losses)
            if k>=kMax:
                return
    



def evaluate(model,root_dir,batchsize=20):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=2)
    with tqdm.tqdm(total=len(d), dynamic_ncols=True,smoothing=0.01) as pbar:
        with torch.no_grad():
            losses = np.zeros(int(len(d)/batchsize))
            accuracies = np.zeros(int(len(d)/batchsize))
            for i,samp in enumerate(dataload):
                x_tensor = samp['image']
                y_tensor = samp['solution']
                y_est = model.forward(x_tensor)
                l = loss(y_est,y_tensor)
                acc = accuracy(y_est,y_tensor)
                losses[i]=l.item()
                accuracies[i] = acc.item()
                pbar.postfix = ',  loss:%0.5f' % np.mean(losses[:(i+1)])
                pbar.update(batchsize)
    return losses,accuracies

## main function
def main(argv):
    N = 1
    try:
        opts = getopt.getopt(argv,"hN:",["N="])[0]
    except getopt.GetoptError:
        print('nn_torch.py -N <# training steps> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('nn_torch.py -N <# training steps> ')
            sys.exit()
        elif opt in ("-N", "--N"):
            N = int(arg)
         
    model = model_deep_class()
    if os.path.isfile('model3_state.pt'):
        model.load_state_dict(torch.load('model3_state.pt'))
    else:
        model.init_weights()
        s = model.state_dict()
        torch.save(s,'model3_state.pt')
        
    optimize_saved(model,N,'training',clip=5,lr=0.0001,momentum=0.5,smooth_display=0.95,loss_file='trainlossD',batchsize=50)
    s = model.state_dict()
    torch.save(s,'model3_state.pt')
    
if __name__== "__main__":
    main(sys.argv[1:])
    #m = model_pred_like()
    #optimize_saved(m,1,'training',lr=0.01,momentum=0,batchsize=8,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=5)
    #pass

## Fiddeling
#model = model_class()
#model.init_weights()
#
#distance= 10
#exponent = 1
#angle = 1
#abs_angle = 1
#im = dl.generate_image(exponent,0,distance,angle,abs_angle,sizes)
#x_test = im[0].transpose(2,0,1)
#y_test = im[3]
#y_test_tensor = torch.tensor([y_test],dtype=torch.float32)
#
#x_test_tensor = torch.tensor([x_test],dtype=torch.float32)
#x2 = model.forward(x_test_tensor)
#print(x2.shape)
#l = loss(x2,y_test_tensor)
#print(l)
#
## optimizer:
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
#
#
## convergence check: Same data every step should converge! (and does)
#x,y = dl.create_training_data(10)
#x_tensor = torch.tensor(x.transpose((0,3,1,2)),dtype=torch.float32)
#y_tensor = torch.tensor(y,dtype=torch.float32)
#for i in range(2000):
#    optimizer.zero_grad()
#    
#    y_est = model.forward(x_tensor)
#    l = loss(y_est,y_tensor)
#    l.backward()
#    optimizer.step()
#    print(l.item())