#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:13:48 2019

@author: heiko
"""

import sys, getopt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tqdm

import matplotlib.pyplot as plt
        

import DeadLeaf as dl

# for data loading
import pandas as pd
from skimage import io #, transform
from torch.utils.data import Dataset, DataLoader

eps = 10**-5

sizes = sizes=np.arange(1,6,dtype=np.float)
imSize = np.array([5,5])



## Model definitions

def init_weights_layer_conv(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

def init_weights_layer_linear(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu')/layer.in_features/layer.out_features)
        layer.bias.data.fill_(0)
        
        

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
        image = (image-127)/128
        solution = self.solutions_df.iloc[idx, 1]
        solution = solution.astype('float32') #.reshape(-1, 1)
        sample = {'image': image, 'solution': solution}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class minimal_gibbs(nn.Module):
    def __init__(self, n_neurons = 10, steps=5):
        super(minimal_gibbs, self).__init__()
        self.conv_enc = nn.Conv2d(1, n_neurons, (3, 3), padding=(1, 1))
        self.conv_mu =  nn.Conv2d(n_neurons, 1, (1, 1))
        self.conv_prec =  nn.Conv2d(n_neurons, 1, (1, 1))
        self.conv_coupling =  nn.Conv2d(n_neurons, 4, (1, 1))
        self.conv_dec1 = nn.Conv2d(1, n_neurons, (1, 1))
        self.conv_dec2 = nn.Conv2d(n_neurons, 1, (3, 3), padding=(1, 1))
        log_var_final = torch.nn.Parameter(data=torch.tensor(0.))
        self.register_parameter('log_var_final',log_var_final)
        self.steps = steps
        self.neighbors = np.array([[1,0],[0,1],[-1,0],[0,-1]],dtype=np.int)
        self.init_weights()
        
    def encode(self,x):
        f = self.conv_enc(x)
        return self.conv_mu(f), F.relu(self.conv_prec(f))+ eps, F.relu(self.conv_coupling(f))
        
    def decode(self,z):
        f = self.conv_dec1(z)
        return self.conv_dec2(f)
    
    def reparametrize(self, mu, prec0, coupling):
        prec = prec0 
        noise = torch.randn_like(prec)
        z = mu + noise/torch.sqrt(prec)
        pad = np.max(np.abs(self.neighbors))
        prec_new = torch.sum(coupling,1,keepdim=True) + prec
        for i in range(self.steps):
            z_pad = F.pad(z,(pad, pad, pad, pad))
            mu_new = prec*mu
            k = 0
            for i_neigh in self.neighbors:
                mu_new += coupling[:,k:(k+1)] * z_pad[:, :,
                                  (pad+i_neigh[0]):(mu.shape[2]+pad+i_neigh[0]),
                                  (pad+i_neigh[1]):(mu.shape[3]+pad+i_neigh[1])]
                k = k + 1
            mu_new = mu_new/prec_new
            noise = torch.randn_like(prec)
            z = mu_new + noise/torch.sqrt(prec_new)
        return z

    def forward(self, x):
        x = x.view(-1,1,imSize[0],imSize[1])
        mu, prec, coupling = self.encode(x)
        z = self.reparametrize(mu, prec, coupling)
        x = self.decode(z)
        return x, mu, -torch.log(prec), coupling, self.log_var_final

    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)


class minimal(nn.Module):
    def __init__(self, n_neurons=10):
        super(minimal, self).__init__()
        self.fc_enc_1_mu = nn.Linear(imSize[0] * imSize[1], n_neurons)
        self.fc_enc_1_std = nn.Linear(imSize[0] * imSize[1], n_neurons)
        self.fc_dec_1 = nn.Linear(n_neurons, imSize[0] * imSize[1])
        log_var_final = torch.nn.Parameter(data=torch.tensor(0.))
        self.register_parameter('log_var_final',log_var_final)
        self.init_weights()
        
    def encode(self,x):
        return self.fc_enc_1_mu(x), self.fc_enc_1_std(x)
        
    def decode(self,z):
        return self.fc_dec_1(z)
    
    def reparametrize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = x.view(-1, imSize[0] * imSize[1])
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu,logvar)
        x = self.decode(z)
        return x, mu, logvar, None, self.log_var_final
    
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear)
        nn.init.zeros_(self.fc_enc_1_std.weight)
        nn.init.constant_(self.fc_enc_1_std.bias,-1)


class basic(nn.Module):
    def __init__(self, n_neurons = 10):
        super(basic, self).__init__()
        self.fc_enc_1 = nn.Linear(imSize[0] * imSize[1], 2*n_neurons)
        self.fc_enc_2_mu = nn.Linear(2*n_neurons, n_neurons)
        self.fc_enc_2_std = nn.Linear(2*n_neurons, n_neurons)
        self.fc_dec_2 = nn.Linear(n_neurons,2*n_neurons)
        self.fc_dec_1 = nn.Linear(2*n_neurons, imSize[0] * imSize[1])
        log_var_final = torch.nn.Parameter(data=torch.tensor(0.))
        self.register_parameter('log_var_final',log_var_final)
        self.init_weights()
        
    def encode(self,x):
        h1 = F.relu(self.fc_enc_1(x))
        return self.fc_enc_2_mu(h1), self.fc_enc_2_std(h1)
        
    def decode(self,z):
        h1 = F.relu(self.fc_dec_2(z))
        return self.fc_dec_1(h1)
    
    def reparametrize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = x.view(-1, imSize[0] * imSize[1])
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu,logvar)
        x = self.decode(z)
        return x, mu, logvar, None, self.log_var_final
    
    def init_weights(self):
        self.apply(init_weights_layer_conv)
        self.apply(init_weights_layer_linear) 
        nn.init.zeros_(self.fc_enc_2_std.weight)

def loss(x_true, x, mu, logvar, log_var_final,alpha=1):
    var_final = log_var_final.exp()
    MSE = loss_MSE(x_true,x,var_final)
    #MSE = 0.5 * torch.sum((x_true.view(-1,x.shape[1])-x).pow(2))/var_final
    KLD = loss_KLD(logvar,mu)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE/x.shape[0] + alpha*KLD/x.shape[0] + 0.5*log_var_final*x.shape[1]

def loss_MSE(x_true,x,var_final):
    MSE = 0.5 * torch.sum((x_true.reshape(-1)-x.view(-1)).pow(2))/var_final
    return MSE

def loss_KLD(logvar,mu):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def rmse(x_true,x):
    RMSE = torch.sqrt(torch.mean((x_true.reshape(-1)-x.view(-1)).pow(2)))
    return RMSE

## function definitions

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
        
        x, mu, logvar, log_var_final = model.forward(x_tensor)
        l = loss(x_tensor,x,mu,logvar,log_var_final)
        l.backward()
        nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        pbar.postfix = '  loss:%0.5f' % l.item()
        pbar.update()
        #if i % 25 == 24:
        #    print(l.item())

def optimize_saved(model,N,root_dir,optimizer,batchsize=20,clip=1,smooth_display=0.9,loss_file=None,kMax=np.inf,smooth_l = 0, device='cpu',alpha = 1):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=10)
    print('starting optimization\n')
    if loss_file:
        if os.path.isfile(loss_file):
            losses = np.load(loss_file)
        else:
            losses = np.array([])
    with tqdm.tqdm(total=min(N*len(d),kMax*batchsize), dynamic_ncols=True,smoothing=0.01) as pbar:
        k0 = len(losses)
        k = k0
        losses = np.concatenate((losses,np.zeros(int(N*len(d)/batchsize))))
        for iEpoch in range(N):
            for i,samp in enumerate(dataload):
                k=k+1
                x_tensor = samp['image'][:,2].to(device)
                #y_tensor = samp['solution'].to(device)
                optimizer.zero_grad()
                x, mu, logvar, coupling, log_var_final = model.forward(x_tensor)
                l = loss(x_tensor,x,mu,logvar,log_var_final,alpha = alpha)
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
                    return

def overtrain(model,root_dir,optimizer,batchsize=20,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=np.inf,smooth_l = 0, device='cpu',alpha=1):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=6)
    print('starting optimization\n')
    with tqdm.tqdm(total=min(len(d),batchsize*kMax), dynamic_ncols=True,smoothing=0.01) as pbar:
        losses = np.zeros(int(len(d)/batchsize))
        k = 0
        for i,samp in enumerate(dataload):
            k=k+1
            if i == 0:
                x_tensor = samp['image'][:,2].to(device)
                y_tensor = samp['solution'].to(device)
            optimizer.zero_grad()
            x, mu, logvar = model.forward(x_tensor)
            l = loss(x_tensor,x,mu,logvar,alpha=alpha)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            smooth_l = smooth_display*smooth_l+(1-smooth_display) * l.item()
            losses[k-1]=l.item()
            pbar.postfix = ',  loss:%0.5f' % (smooth_l/(1-smooth_display**k))
            pbar.update(batchsize)
            if k>=kMax:
                return

def evaluate(model,root_dir,batchsize=20, device='cpu',alpha=1):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=batchsize,shuffle=True,num_workers=2)
    with tqdm.tqdm(total=len(d), dynamic_ncols=True,smoothing=0.01) as pbar:
        with torch.no_grad():
            losses = np.zeros(int(len(d)/batchsize))
            MSEs = np.zeros(int(len(d)/batchsize))
            KLDs = np.zeros(int(len(d)/batchsize))
            accuracies = np.zeros((int(len(d)/batchsize),2))
            for i,samp in enumerate(dataload):
                x_tensor = samp['image'].to(device)
                #y_tensor = samp['solution'].to(device)
                x, mu, logvar, coupling, log_var_final = model.forward(x_tensor)
                l = loss(x_tensor,x,mu,logvar,log_var_final,alpha=1)
                MSE = loss_MSE(x_tensor,x,log_var_final.exp())/x.shape[0]
                KLD = loss_KLD(logvar,mu)/x.shape[0]
                acc = rmse(x_tensor,x)
                x_reconstruct = model.decode(mu)
                acc2 = rmse(x_tensor,x_reconstruct)
                losses[i]=l.item()
                KLDs[i]=KLD.item()
                MSEs[i]=MSE.item()
                accuracies[i] = [acc.item(),acc2.item()]
                pbar.postfix = ',  loss:%0.5f' % np.mean(losses[:(i+1)])
                pbar.update(batchsize)
    return losses,accuracies,MSEs,KLDs

def count_positive(root_dir):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=20,shuffle=True,num_workers=6)
    pos_samples = 0
    all_samples = 0
    for i,samp in enumerate(dataload):
        pos_samples = pos_samples + np.sum(samp['solution'].detach().numpy())
        all_samples = all_samples + len(samp['solution'].detach().numpy())
    return pos_samples,all_samples

def show(model,root_dir,n_image=5):
    d = dead_leaves_dataset(root_dir)
    dataload = DataLoader(d,batch_size=n_image,shuffle=False,num_workers=1)
    samp = next(iter(dataload))
    x_true = samp['image'][:,2]
    x, mu, logvar, coupling, log_var_final = model(x_true)
    x = x.reshape(-1,5,5)
    x_reconstruct = model.decode(mu)
    x_reconstruct = x_reconstruct.reshape(-1,5,5)
    
    plt.figure(figsize=(2.5,5))
    for i_image in range(n_image):
        plt.subplot(n_image,3,3*i_image+1)
        plt.imshow(x_true[i_image],cmap=plt.gray(),clim=[-1,1])
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False)
        
        plt.subplot(n_image,3,3*i_image+2)
        plt.imshow(x_reconstruct[i_image].detach().cpu().numpy(),cmap=plt.gray(),clim=[-1,1])
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False)
        if i_image ==0:
            plt.title('reconstructed')
        
        plt.subplot(n_image,3,3*i_image+3)
        plt.imshow(x[i_image].detach().cpu().numpy(),cmap=plt.gray(),clim=[-1,1])
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False)
        if i_image ==0:
            plt.title('sampled')
    plt.tight_layout(pad=0.1)
    plt.figure(figsize=(7.5,2.5))
    plt.subplot(1,2,1)
    plt.plot(mu.reshape(n_image,-1).detach().numpy().T,'k')
    plt.ylabel('Mean')
    plt.xlabel('z-dimension')
    plt.subplot(1,2,2)
    plt.plot(logvar.reshape(n_image,-1).detach().numpy().T,'k')
    plt.ylabel('log Variance')
    plt.xlabel('z-dimension')
    plt.tight_layout()
    plt.show()

def main(model_name,action,average_neighbors=False,device='cpu',weight_decay = 10**-3,epochs=1,lr = 0.001,kMax=np.inf,batchsize=20,time=5,n_neurons=10,kernel=3,alpha = 1):
    filename = 'vae_tiny_%s' % model_name
    if model_name == 'basic':
        model = basic(n_neurons).to(device)
    elif model_name == 'min':
        model = minimal(n_neurons).to(device)
    elif model_name == 'min_gibbs':
        model = minimal_gibbs(n_neurons).to(device)
    if not n_neurons==10:
        filename = filename + '_nn%02d' % n_neurons
    if not kernel==3:
        filename = filename + '_k%02d' % kernel
    if not time==5:
        filename = filename + '_%02d' % time
    path= '/Users/heiko/tinytinydeadrects/models/' + filename + '.pt'
    path_opt= '/Users/heiko/tinytinydeadrects/models/' + filename + '_opt.pt'
    path_loss= '/Users/heiko/tinytinydeadrects/models/' + filename + '_loss.npy'
    path_l= '/Users/heiko/tinytinydeadrects/models/' + filename + '_l.npy'
    path_acc= '/Users/heiko/tinytinydeadrects/models/' + filename + '_acc.npy'
    path_MSE= '/Users/heiko/tinytinydeadrects/models/' + filename + '_MSE.npy'
    path_KLD= '/Users/heiko/tinytinydeadrects/models/' + filename + '_KLD.npy'
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
        optimize_saved(model,epochs,data_folder,optimizer,batchsize=batchsize,clip=np.inf,smooth_display=0.9,loss_file=path_loss,kMax=kMax,device=device,alpha=alpha)
        torch.save(model.state_dict(),path)
        torch.save(optimizer.state_dict(),path_opt)
        return model
    elif action =='eval': 
        data_folder = '/Users/heiko/tinytinydeadrects/validation'
        l,acc,MSE,KLD = evaluate(model,data_folder,batchsize=batchsize,device=device,alpha=alpha)
        np.save(path_l,np.array(l))
        np.save(path_acc,np.array(acc))
        np.save(path_MSE,np.array(MSE))
        np.save(path_KLD,np.array(KLD))
        return acc,l
    elif action =='overtrain':
        data_folder = '/Users/heiko/tinytinydeadrects/training'
        overtrain(model,data_folder,optimizer,batchsize=batchsize,clip=np.inf,smooth_display=0.9,loss_file=None,kMax=kMax,device=device,alpha=alpha)
    elif action == 'print':
        print(filename)
        if os.path.isfile(path_l):
            print('total loss:')
            print(np.mean(np.load(path_l)))
            print('RMSE:')
            print(np.mean(np.load(path_acc),axis=0))
            print('MSE-loss:')
            print(np.mean(np.load(path_MSE),axis=0))
            print('KLD-loss:')
            print(np.mean(np.load(path_KLD),axis=0))
            print('sigma estimate:')
            print(np.exp(0.5*model.log_var_final.detach().numpy()))
        else:
            print('not yet evaluated!')
    elif action == 'show':
        data_folder = '/Users/heiko/tinytinydeadrects/training'
        show(model, data_folder)
    

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
    parser.add_argument("-a","--alpha",type=float,help="weight for the KL divergence",default=1)
    parser.add_argument("--kernel",type=int,help="kernel size",default=3)
    parser.add_argument("action",help="what to do? [train,eval,overtrain,print,reset,show]", choices=['train', 'eval', 'overtrain', 'print', 'reset', 'show'])
    parser.add_argument("model_name",help="model to be trained [min,basic]", choices=['min','basic','min_gibbs'])
    args=parser.parse_args()
    main(args.model_name,args.action,device=args.device,weight_decay=float(args.weight_decay),epochs=args.epochs,
         lr = args.learning_rate,kMax=args.kMax,batchsize=args.batch_size,time=args.time,n_neurons=args.n_neurons,kernel=args.kernel,alpha=args.alpha)