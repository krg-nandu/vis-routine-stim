#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:46:47 2018

@author: heiko
"""


import numpy as np
import DeadLeaf as dl
import matplotlib.pyplot as plt


distances = np.arange(0,150)

sizes = 5*np.array([np.arange(1,80,dtype='float'),np.arange(1,80,dtype='float')])
prob = (sizes[0]/np.min(sizes[0])) **-3
sizes = sizes.transpose()

p1 = [dl.calc_prob_one(sizes=sizes,prob=prob,dx=k,dy=0) for k in distances]
p2d = [[dl.calc_prob_one(sizes=[s],prob=None,dx=k,dy=0) for k in distances] for s in sizes]

plt.figure()
ax = plt.subplot()
plt.plot(np.transpose(p2d))
plt.plot(p1,'k',linewidth=2)
plt.xlabel('distance [px]')
plt.ylabel('P(same object)')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('PriorVsDistanceSquare.pdf')

sizes = 5*np.arange(1,80,dtype='float')
prob = (sizes/np.min(sizes)) **-1.5

p1 = [dl.calc_prob_one(sizes=sizes,prob=prob,dx=k,dy=0) for k in distances]
p2d = [[dl.calc_prob_one(sizes=[s],prob=None,dx=k,dy=0) for k in distances] for s in sizes]

print('Sizes chosen for horizontal:')
print(np.argmax(np.array(p1)<.9), p1[np.argmax(np.array(p1)<.9)])
print(np.argmax(np.array(p1)<.75),p1[np.argmax(np.array(p1)<.75)])
print(np.argmax(np.array(p1)<.5), p1[np.argmax(np.array(p1)<.5)])
print(np.argmax(np.array(p1)<.25),p1[np.argmax(np.array(p1)<.25)])
print(np.argmax(np.array(p1)<.1), p1[np.argmax(np.array(p1)<.1)])

plt.figure()
ax = plt.subplot()
plt.plot(np.transpose(p2d))
plt.plot(p1,'k',linewidth=2)
plt.xlabel('distance [px]')
plt.ylabel('P(same object)')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('PriorVsDistanceRect.pdf')


p1 = [dl.calc_prob_one(sizes=sizes,prob=prob,dx=k,dy=k) for k in distances]
p2d = [[dl.calc_prob_one(sizes=[s],prob=None,dx=k,dy=k) for k in distances] for s in sizes]


print('Sizes chosen for diagonal:')
print(np.argmax(np.array(p1)<.9), p1[np.argmax(np.array(p1)<.9)])
print(np.argmax(np.array(p1)<.75),p1[np.argmax(np.array(p1)<.75)])
print(np.argmax(np.array(p1)<.5), p1[np.argmax(np.array(p1)<.5)])
print(np.argmax(np.array(p1)<.25),p1[np.argmax(np.array(p1)<.25)])
print(np.argmax(np.array(p1)<.1), p1[np.argmax(np.array(p1)<.1)])

plt.figure()
ax = plt.subplot()
plt.plot(np.transpose(p2d))
plt.plot(p1,'k',linewidth=2)
plt.xlabel('distance [px]')
plt.ylabel('P(same object)')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('PriorVsDistanceDiagonal.pdf')


p1 = [dl.calc_prob_one(sizes=sizes,prob=None,dx=k,dy=0) for k in distances]
p2d = [[dl.calc_prob_one(sizes=[s],prob=None,dx=k,dy=0) for k in distances] for s in sizes]

plt.figure()
ax = plt.subplot()
plt.plot(np.transpose(p2d))
plt.plot(p1,'k',linewidth=2)
plt.xlabel('distance [px]')
plt.ylabel('P(same object)')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('PriorVsDistanceFlat.pdf')