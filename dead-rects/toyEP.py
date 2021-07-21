#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:51:53 2019

@author: heiko
"""
import numpy as np



# Sigma is a variance here!
def norm_prod_gauss(mu1,mu2,sigma1,sigma2):
    mu = sigma2/(sigma1+sigma2)*mu1 + sigma1/(sigma1+sigma2)*mu2
    sigma = 1/(1/sigma1+1/sigma2)
    norm = norm_pdf(mu,mu1,sigma1)*norm_pdf(mu,mu2,sigma2)/norm_pdf(mu,mu,sigma)
    return mu,sigma,norm

def norm_prod_gauss_mult(mu1,mu2,prec1,prec2):
    prec = prec1+prec2
    mu1 = mu1.reshape((prec.shape[0],1))
    mu2 = mu2.reshape((prec.shape[0],1))
    mu = np.linalg.solve(prec,np.matmul(prec1,mu1)+np.matmul(prec2,mu2))
    norm = norm_pdf_mult(mu,mu2,prec2)*norm_pdf_mult(mu,mu1,prec1)/norm_pdf_mult(mu,mu,prec)
    return mu,prec,norm

def norm_pdf(x,mu,sigma):
    return np.exp(-(x-mu)*(x-mu)/2/sigma)/np.sqrt(2*np.pi*sigma)

def norm_pdf_mult(x,mu,prec):
    assert prec.shape[0] == prec.shape[1]
    mu = mu.reshape((prec.shape[0],1))
    x = x.reshape((prec.shape[0],1))
    return np.exp(-0.5*np.matmul((x-mu).T,np.matmul(prec,(x-mu))))/((2*np.pi)**(prec.shape[0]/2))*np.sqrt(np.linalg.det(prec))

prior_s1 = 1
prior_s2 = 1

s1 = -20
s2 = 1.1
c12 = 5

sigma = 0.05
pobs = 1/sigma

### Variational algorithm
mu1 = 0
mu2 = 0

prec1 = np.array([prior_s1 + 1/sigma])
prec2 = np.array([prior_s2 + 1/sigma])

mu1 = s1/sigma/prec1
mu2 = s2/sigma/prec2

w12 = 0

wrecord = []
mrecord = []
srecord = []


for i in range(10):
    prec1p = prec1-w12*c12-pobs
    m1pri = (prec1*mu1-w12*c12*mu2-pobs*s1)/prec1p
    prec2p = prec2-w12*c12-pobs
    m2pri = (prec2*mu2-w12*c12*mu1-pobs*s2)/prec2p
    
    # Likelihood parameters
    mL = np.array([s1,s2])
    pL = np.array([[pobs,0],[0,pobs]])
    
    # parameters for L=0
    m0 = np.array([m1pri,m2pri])
    p0 = np.array([np.concatenate((prec1p,np.array([0]))),np.concatenate((np.array([0]),prec2p))])
    m0post,p0post,norm0 = norm_prod_gauss_mult(m0,mL,p0,pL)
    
    # parameters for L=1
    p1 = np.array([[prec1p+c12,-c12],[-c12,prec2p+c12]])
    p1 = np.array([np.concatenate((prec1p+c12,np.array([-c12]))),np.concatenate((np.array([-c12]),prec2p+c12))])
    m1 = np.linalg.solve(p1,np.matmul(p0,m0))
    m1post,p1post,norm1 = norm_prod_gauss_mult(m1,mL,p1,pL)
    
    # update w12 = P(L12=1)
    w12 = norm1.flatten()/(norm1.flatten()+norm0.flatten())
    
    # Update features
    prec1 = prec1p + pobs + w12*c12
    mu1new = (prec1p*m1pri+pobs*s1+w12*c12*mu2)/prec1
    prec2 = prec2p + pobs + w12*c12
    mu2new = (prec2p*m2pri+pobs*s2+w12*c12*mu1)/prec2
    mu1 = mu1new
    mu2 = mu2new
    
    wrecord.append(w12)
    mrecord.append(mu1)
    srecord.append(prec1)
wrecord = np.array(wrecord)
mrecord = np.array(mrecord)
srecord = np.array(srecord)
    

### EP algorithm
mu1 = 0
mu2 = 0

sigma1 = prior_s1
sigma2 = prior_s2

w12 = 0.1

wrecord = []
mrecord = []
srecord = []

Ainv = np.array([[1/prior_s1,0],[0,1/prior_s2]])
a = np.array([0,0])
za = 1/np.sqrt(np.linalg.det(Ainv))*np.exp(-0.5*np.matmul(np.matmul(a.T,Ainv),a))
Cinv = np.array([[1/prior_s1+1/c12,-1/c12],[-1/c12,1/prior_s2+1/c12]])
c = np.linalg.solve(Cinv,[mu1/sigma1,mu2/sigma2])
zc = 1/np.sqrt(np.linalg.det(Cinv))*np.exp(-0.5*np.matmul(np.matmul(a.T,Cinv),a))
norm_prior = za/zc

for i in range(10):
    # update 1
    mu1_no_coupling,sigma1_no_coupling,norm1_no_coupling = norm_prod_gauss(s1,0,sigma,prior_s1)
    mu1_coupling,sigma1_coupling,norm1_coupling = norm_prod_gauss(mu2,mu1_no_coupling,sigma2+c12,sigma1_no_coupling)
    w = (w12*norm1_coupling)/(1-w12 + w12*norm1_coupling)
    mu1 = (1-w)*mu1_no_coupling + w * mu1_coupling
    sigma1 = w*(sigma1_no_coupling+(mu1_no_coupling-mu1)**2) + (1-w)*(sigma1_coupling+(mu1_coupling-mu1)**2)
    # update 2
    mu2_no_coupling,sigma2_no_coupling,norm2_no_coupling = norm_prod_gauss(s2,0,sigma,prior_s2)
    mu2_coupling,sigma2_coupling,norm2_coupling = norm_prod_gauss(mu1,mu2_no_coupling,sigma1+c12,sigma2_no_coupling)
    w = (w12*norm2_coupling)/(1-w12 + w12*norm2_coupling)
    mu2 = (1-w)*mu2_no_coupling + w * mu2_coupling
    sigma2 = w*(sigma2_no_coupling+(mu2_no_coupling-mu2)**2) + (1-w)*(sigma2_coupling+(mu2_coupling-mu2)**2)
    # w12
    #_,_,raw_w0 = norm_prod_gauss(mu1,mu2,sigma1,sigma2)
    #m,sig,raw_w11 = norm_prod_gauss(mu1,mu2,sigma1,c12)
    #_,_,raw_w12 = norm_prod_gauss(m,mu2,sigma1,c12)
    # get 2D Gaussian:
    Ainv = np.array([[1/sigma1,0],[0,1/sigma2]])
    a = np.array([mu1,mu2])
    za = 1/np.sqrt(np.linalg.det(Ainv))*np.exp(-0.5*np.matmul(np.matmul(a.T,Ainv),a))
    Cinv = np.array([[1/sigma1+1/c12,-1/c12],[-1/c12,1/sigma2+1/c12]])
    c = np.linalg.solve(Cinv,[mu1/sigma1,mu2/sigma2])
    zc = 1/np.sqrt(np.linalg.det(Cinv))*np.exp(-0.5*np.matmul(np.matmul(a.T,Cinv),a))
    w12 = zc/(za+zc)
    wrecord.append(w12)
    mrecord.append(mu1)
    srecord.append(sigma1)
wrecord = np.array(wrecord)
mrecord = np.array(mrecord)
srecord = np.array(srecord)