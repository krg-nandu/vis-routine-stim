#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:12:51 2019

@author: heiko
"""


import numpy as np
import os
import tqdm
import time
import PIL
import scipy.ndimage as ndimage
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# put these into the library at some timepoint!

folder= 'resultsFrozen'
distances = [5,10,20,40,80]
distancesd = [4,7,14,28,57]

#against distance
dirs = os.listdir(folder)
#dirs.remove('test')
dirs.remove('.DS_Store')
dirs.remove('Heiko')
dirs.remove('Icon\r')
k =0
while k<len(dirs):
    if 'train' in dirs[k]:
        dirs.remove(dirs[k])
    elif 'test' in dirs[k]:
        dirs.remove(dirs[k])
    else:
        k = k + 1

dirs.sort()

d = np.zeros((0,10))
k =-1
for iDir in range(len(dirs)):
    k=k+1
    files = os.listdir(folder+'/'+dirs[iDir])
    for iFile in range(len(files)):
        if files[iFile][0:6]=='result':
            dnew = np.load(folder+'/'+dirs[iDir]+'/'+files[iFile])
            d = np.append(d,np.hstack([k*np.ones((dnew.shape[0],1)),dnew]),axis=0)
            
df = pd.DataFrame(d,columns=('subjN','exponent','num_cols','distance','angle','abs_angle','imageN','solution','response','visible_solution'))
df = df.assign(subject=df['subjN'].astype('category'))
N = len(np.unique(df['subject']))
df = df.assign(correct=(df['response']==df['solution']))
df = df.assign(choice=(df['response']>=0))
df = df.assign(truth=(df['solution']>=0))

df_subjects = df.groupby('subject')
bias_summary=df_subjects['response'].describe().reset_index()
correct_summary = df_subjects['correct'].describe().reset_index()

df_subjects = df.groupby('subject').agg(np.mean)


df_image=df.groupby(['exponent','distance','angle','abs_angle','imageN']).agg(np.mean)

np.save('image_summary.npy',df_image.values)


sns.set()
sns.set_context("talk",rc={"lines.linewidth": 2.5,'lines.markeredgewidth': 1,'lines.markersize': 10},font_scale=1.25)

plt.figure(figsize=(15,5))

ax = plt.subplot(1,2,1)
plt.plot(np.arange(1,6),df.groupby('exponent')['truth'].aggregate(np.mean),'ko-')
sns.lineplot(data=df,ax=ax,x='exponent',y='choice',marker='s',err_style=None,color=(0,0.25,0.25))
sns.lineplot(data=df,ax=ax,x='exponent',y='choice',hue='subject',err_style=None,palette=N*[(0,0.8,0.8)],legend=None)
sns.lineplot(data=df,ax=ax,x='exponent',y='choice',marker='s',err_style=None,color=(0,0.25,0.25))
plt.plot(np.arange(1,6),df.groupby('exponent')['truth'].aggregate(np.mean),'ko-')

plt.ylabel('Proportion same reports')
plt.xlabel(r'Exponent $\alpha$, size distribution')
plt.ylim([0,1])
plt.xlim([0,6])
ax.set_xticks([1,2,3,4,5])
plt.legend(['truth','mean','subj'],frameon=False)

ax = plt.subplot(1,2,2)
sns.lineplot(data=df,ax=ax,x='exponent',y='correct',hue='subject',err_style=None,palette=N*[(0,0.8,0.8)],legend=None)
sns.lineplot(data=df,ax=ax,x='exponent',y='correct',marker='s',err_style=None,color=(0,0.25,0.25))
plt.plot([0,6],[0.5,0.5],'k:')

plt.ylabel('Proportion correct reports')
plt.xlabel(r'Exponent $\alpha$, size distribution')
plt.ylim([.4,1])
plt.xlim([0,6])
ax.set_xticks([1,2,3,4,5])

plt.figure(figsize=(15,5))

ax = plt.subplot(1,2,1)
sns.lineplot(data=df,ax=ax,x='distance',y='choice',marker='s',err_style=None,palette=[(0,0,0.8),(0,0.8,0)],hue='angle',legend=None)
sns.lineplot(data=df[df.angle==0],ax=ax,x='distance',y='choice',hue='subject',err_style=None,palette=N*[(0.7,0.7,1)],legend=None)
sns.lineplot(data=df[df.angle==1],ax=ax,x='distance',y='choice',hue='subject',err_style=None,palette=N*[(0.7,1,0.7)],legend=None)
sns.lineplot(data=df,ax=ax,x='distance',y='choice',marker='s',err_style=None,palette=[(0,0,0.8),(0,0.8,0)],hue='angle',legend=None)
plt.plot(df[df.angle==0].groupby('distance')['distance'].aggregate(np.mean),df[df.angle==0].groupby('distance')['truth'].aggregate(np.mean),'o-',color=(0,0,0.3))
plt.plot(df[df.angle==1].groupby('distance')['distance'].aggregate(np.mean),df[df.angle==1].groupby('distance')['truth'].aggregate(np.mean),'o-',color=(0,0.3,0))

plt.ylabel('Proportion same reports')
plt.xlabel('Distance [px]')
plt.ylim([0,1])
plt.xlim([0,85])
ax.set_xticks(distances)

ax = plt.subplot(1,2,2)
sns.lineplot(data=df[df.angle==0],ax=ax,x='distance',y='correct',marker='s',err_style=None,color=(0,0,0.8),legend=None)
sns.lineplot(data=df[df.angle==1],ax=ax,x='distance',y='correct',marker='s',err_style=None,color=(0,0.8,0),legend=None)
sns.lineplot(data=df[df.angle==0],ax=ax,x='distance',y='correct',hue='subject',err_style=None,palette=N*[(0.7,0.7,1)],legend=None)
sns.lineplot(data=df[df.angle==1],ax=ax,x='distance',y='correct',hue='subject',err_style=None,palette=N*[(0.7,1,0.7)],legend=None)
sns.lineplot(data=df[df.angle==0],ax=ax,x='distance',y='correct',marker='s',err_style=None,color=(0,0,0.8),legend=None)
sns.lineplot(data=df[df.angle==1],ax=ax,x='distance',y='correct',marker='s',err_style=None,color=(0,0.8,0),legend=None)
plt.plot([0,85],[0.5,0.5],'k:')

plt.ylabel('Proportion correct reports')
plt.xlabel('Distance [px]')
plt.ylim([.4,1])
plt.xlim([0,85])
ax.set_xticks(distances)
plt.legend(['cardinal','diagonal'],frameon=False,loc='upper right')

# This is the only way I found to do this. It looks like a HORRIBLE IDEA
# but unfortunately it works so far.
#import sys
#sys.path.append('..')
#import DeadLeaf as dl
#sys.path.pop(-1)
#
df_hard = df_image[df_image['correct']<0.1]
df_hard.reset_index(inplace=True)
#for index, row in df_hard.iterrows():
   #print(row['exponent'])
#   dl.show_frozen_image(exponent= row['exponent'],dist=row['distance'],angle=row['angle'],abs_angle=row['abs_angle'],i=row['imageN'],border=1)
   
   
## Histograms:
df_image=df.groupby(['exponent','distance','angle','abs_angle','imageN']).agg(np.sum)
df_image.reset_index(inplace=True)
   
with sns.axes_style("white"):
    plt.figure(figsize=(15,5))
    ax = plt.subplot(1,2,1)
    sns.distplot(df_image['correct'], kde=False,bins=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-0.5)
    plt.xlabel('# subjects judging correctly')
    plt.ylabel('# of images')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim([0,400])
    #x = np.arange(15)
    #plt.plot(x,2000*stats.binom.pmf(x,14,np.mean(df_image['correct']/14)),'k')
           
    ax = plt.subplot(1,2,2)
    sns.distplot(df_image['choice'], kde=False,bins=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-0.5)
    plt.xlabel('# subjects judging \'same\'')
    plt.ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim([0,400])
    #plt.plot(x,2000*stats.binom.pmf(x,14,np.mean(df_image['choice']/14)),'k')
           