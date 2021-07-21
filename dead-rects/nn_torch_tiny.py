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

sizes = sizes=5*np.arange(1,6,dtype=np.float)
imSize = np.array([30,30])
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
        
def activity_one_regularizer(activities):
    return torch.mean((activities-1) * (activities-1))
          
class model_min_class(nn.Module):
    def __init__(self):
        super(model_min_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.fc = nn.Linear(3*30*30, 1)
    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 3 * 30 * 30)
        x = self.fc(x)
        x = torch.exp(x)/(torch.exp(x)+1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_class(nn.Module):
    def __init__(self):
        super(model_class, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
    def forward(self, x):
        x = self.norm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.exp(x)/(torch.exp(x)+1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)

class model_residual(nn.Module):
    def __init__(self):
        super(model_residual, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5,padding=(2,2))
        self.conv1_1 = nn.Conv2d(10, 10, kernel_size=5,padding=(2,2))
        self.conv1_2 = nn.Conv2d(10, 10, kernel_size=5,padding=(2,2))
        self.conv1_3 = nn.Conv2d(10, 10, kernel_size=5,padding=(2,2))
        self.conv1_4 = nn.Conv2d(10, 10, kernel_size=5,padding=(2,2))
        self.conv1_5 = nn.Conv2d(10, 10, kernel_size=5,padding=(2,2))
        self.conv1_6 = nn.Conv2d(10, 10, kernel_size=5,padding=(2,2))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5,padding=(2,2))
        self.conv2_1 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv2_2 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv2_3 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv2_4 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv2_5 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv2_6 = nn.Conv2d(20, 20, kernel_size=5,padding=(2,2))
        self.conv3 = nn.Conv2d(20, 30, 5, padding=(2,2))
        self.fc1 = nn.Linear(30*7*7, 120)
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
        x3 = self.conv3(self.pool(x2_2))
        x = x3.view(-1, 30 * 7 * 7)
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
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5,stride=2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3 = nn.Conv2d(40, 100, 5)
        self.conv4 = nn.Conv2d(100, 200, 5)
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)
    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 200)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.exp(x)/(torch.exp(x)+1)
        return x
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class model_recurrent(nn.Module):
    def __init__(self,Nrep=20):
        super(model_recurrent, self).__init__()
        self.norm = nn.InstanceNorm2d(3)
        self.norm1 = nn.InstanceNorm2d(20)
        self.norm2 = nn.InstanceNorm2d(40)
        self.conv1 = nn.Conv2d(23, 20, kernel_size=5,stride=1,padding=(2,2))
        self.conv2 = nn.Conv2d(60, 40, kernel_size=5,stride=1,padding=(2,2))
        self.pool = nn.MaxPool2d(5, 5)
        self.pool6 = nn.MaxPool2d(6, 6)
        self.fc1 = nn.Linear(40*6*6, 200)
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
            inp1 = torch.cat((x,self.norm1(h1)),dim=1)
            h1 = F.relu(self.conv1(inp1))
            inp2 = torch.cat((self.pool(h1),self.norm2(h2)),dim=1)
            h2 = F.relu(self.conv2(inp2))
        #o2 = self.pool6(h2)
        x = h2.view(-1, 40 *6*6)
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
        x = out2.view(-1, 10)
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

def optimize_saved(model,N,root_dir,optimizer,batchsize=20,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=np.inf,smooth_l = 0):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=6)
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

def overtrain(model,root_dir,optimizer,batchsize=20,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=np.inf,smooth_l = 0):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=6)
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

def count_positive(root_dir):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=20,shuffle=True,num_workers=6)
    pos_samples = 0
    all_samples = 0
    for i,samp in enumerate(dataload):
        pos_samples = pos_samples + np.sum(samp['solution'].detach().numpy())
        all_samples = all_samples + len(samp['solution'].detach().numpy())
    return pos_samples,all_samples

def main(model_name,action,average_neighbors=False,device='cpu',weight_decay = 10**-3,epochs=1,lr = 0.001,kMax=np.inf,batchsize=20):
    filename = 'model_tiny_%s' % model_name
    if model_name == 'model':
        model = model_class().to(device)
    elif model_name == 'deep':
        model = model_deep_class().to(device)
    elif model_name == 'pred':
        model = model_pred_like().to(device)
    elif model_name == 'recurrent':
        model = model_recurrent().to(device)
    elif model_name == 'min':
        model = model_min_class().to(device)
    elif model_name == 'res':
        model = model_residual().to(device)
    path= '/Users/heiko/tinydeadrects/models/' + filename + '.pt'
    path_opt= '/Users/heiko/tinydeadrects/models/' + filename + '_opt.pt'
    path_loss= '/Users/heiko/tinydeadrects/models/' + filename + '_loss.npy'
    path_l= '/Users/heiko/tinydeadrects/models/' + filename + '_l.npy'
    path_acc= '/Users/heiko/tinydeadrects/models/' + filename + '_acc.npy'
    optimizer = torch.optim.Adam(model.parameters(),lr = lr,amsgrad=True,weight_decay=weight_decay)
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
        data_folder = '/Users/heiko/tinydeadrects/training'
        optimize_saved(model,epochs,data_folder,optimizer,batchsize=batchsize,clip=np.inf,smooth_display=0.9,loss_file=path_loss,kMax=kMax)
        torch.save(model.state_dict(),path)
        torch.save(optimizer.state_dict(),path_opt)
        return model
    elif action =='eval': 
        data_folder = '/Users/heiko/tinydeadrects/validation'
        l,acc = evaluate(model,data_folder,batchsize=batchsize)
        np.save(path_l,np.array(l))
        np.save(path_acc,np.array(acc))
        return acc,l
    elif action =='overtrain':
        data_folder = '/Users/heiko/tinydeadrects/training'
        overtrain(model,data_folder,optimizer,batchsize=batchsize,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=kMax)
    elif action == 'print':
        print(filename)
        if os.path.isfile(path_l):
            print('negative log-likelihood:')
            print(np.mean(np.load(path_l)))
            print('accuracy:')
            print(np.mean(np.load(path_acc)))
        else:
            print('not yet evaluated!')
    

### Testing the evaluation for imagenet
#mTest = minimumNet().to(device)
#mEvalTest,accTest,lTest = evaluate_acc(mTest,data_loader,val_loader,device=device,maxiter=np.inf)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--device", help="device to run on [cuda,cpu]", choices=['cuda', 'cpu'],default='cpu')
    parser.add_argument("-E","--epochs", help="numer of epochs", type = int ,default=1)
    parser.add_argument("-b","--batch_size", help="size of a batch", type = int ,default=20)
    parser.add_argument("-w","--weight_decay",type=float,help="how much weight decay?",default=10**-3)
    parser.add_argument("-l","--learning_rate",type=float,help="learning rate",default=10**-3)
    parser.add_argument("-k","--kMax",type=int,help="maximum number of training steps",default=np.inf)
    parser.add_argument("action",help="what to do? [train,eval,overtrain,print,reset]", choices=['train', 'eval', 'overtrain', 'print', 'reset'])
    parser.add_argument("model_name",help="model to be trained [model,deep,recurrent,pred]", choices=['model', 'deep','res','recurrent','pred','min'])
    args=parser.parse_args()
    main(args.model_name,args.action,device=args.device,weight_decay=float(args.weight_decay),epochs=args.epochs,lr = args.learning_rate,kMax=args.kMax,batchsize=args.batch_size)