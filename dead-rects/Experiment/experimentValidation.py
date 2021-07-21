#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:58:28 2018

@author: Heiko
"""


from psychopy import visual,core,event,monitors
import numpy as np
import PIL
import pickle
import os
import pandas as pd
from scipy.ndimage import convolve1d as conv

# This is the only way I found to do this. It looks like a HORRIBLE IDEA
# but unfortunately it works so far.
import sys
sys.path.append('..')
import DeadLeaf as dl
sys.path.pop(-1)

import datetime

monitors.monitorFolder= '/Users/heiko/.psychopy3/monitors'
display_size = np.array([900,900])

name = input("What is your subject code?")
res_folder = './resultsValidation/%s/' % name
n_trials=input('How many trials do you want to do today?')
offset_image_number = input('How far shall I offset the image number?')
im_size = input('What size shall the images have?')
im_size = int(im_size)
im_folder = '/Users/heiko/deadrects/validation_%d/' % im_size

solutions_df = pd.read_csv(os.path.join(im_folder,'solution.csv'),index_col=0)

if not os.path.exists(res_folder):
    os.mkdir(res_folder)

n_trials = int(n_trials) # per condition
offset_image_number = int(offset_image_number)
n_cols = np.array([9])
exponents = np.arange(5)+1
exponents = exponents[np.random.permutation(len(exponents))]

core.checkPygletDuringWait = True

#Setup for Screen and 
# Create the window object.
window = visual.Window(fullscr=True,monitor='Experiment1',units='pix',wintype='pyglet',screen=0)
clock = window.frameClock
event.clearEvents()
#
#core.wait(1,1)
#

#CenterImage.setImage('../sizeInvariantSquares.png')
#CenterImage.draw(window)
#window.flip()
#core.wait(1,1)
    




def Trial(window,clock,imageNumber=0): 
    im_name = im_folder + 'image%07d.png' % imageNumber
    image = PIL.Image.open(im_name)
    if im_size == 300:
        im = np.array(image)
        im[:,:,0] = conv(im[:,:,0],np.ones(3),axis=0)
        im[:,:,0] = conv(im[:,:,0],np.ones(3),axis=1)
        image = PIL.Image.fromarray(im)
    t0 = window.flip()
    CenterImage = visual.ImageStim(window)
    im_pil = image.resize(display_size)
    CenterImage.setImage(im_pil)
    CenterImage.draw(window)
    t1 = window.flip()
    keypresses = event.waitKeys(maxWait=10,timeStamped=clock)
    t2 = window.flip()
    sol = solutions_df['solution'][imageNumber]
    return keypresses, sol, (t0,t1,t2)

def draw_pos_marker(window,pos,lw=3):
    l1 = visual.Line(window,start=pos+[-15.5,0.5],end=pos+[-3.5,0.5],lineColor='red', lineWidth=lw)
    l2 = visual.Line(window,start=pos+[ 15.5,0.5],end=pos+[ 3.5,0.5],lineColor='red', lineWidth=lw)
    l3 = visual.Line(window,start=pos+[0.5,-15.5],end=pos+[0.5,-3.5],lineColor='red', lineWidth=lw)
    l4 = visual.Line(window,start=pos+[0.5, 15.5],end=pos+[0.5, 3.5],lineColor='red', lineWidth=lw)
    l1.draw(window)
    l2.draw(window)
    l3.draw(window)
    l4.draw(window)
    
def draw_pos_marker_diagonal(window,pos,lw=3):
    l1 = visual.Line(window,start=pos-14.5,end=pos-2.5,lineColor='red', lineWidth=lw)
    l2 = visual.Line(window,start=pos+15.5,end=pos+3.5,lineColor='red', lineWidth=lw)
    l3 = visual.Line(window,start=pos+[15.5,-14.5],end=pos+[3.5,-2.5],lineColor='red', lineWidth=lw)
    l4 = visual.Line(window,start=pos+[-14.5,15.5],end=pos+[-2.5,3.5],lineColor='red', lineWidth=lw)
    l1.draw(window)
    l2.draw(window)
    l3.draw(window)
    l4.draw(window)
    


def draw_pos_marker_dot(window,pos,lw=5):
    #p = visual.Circle(window, radius=lw,pos = pos,lineColor='red')
    #p = visual.Line(window,start=pos,end=pos,lineColor='red', lineWidth=lw)
    p = visual.Rect(window,pos = pos+np.array([0.5,0.5]),width=lw,height=lw,fillColor='red',lineColor='red',lineWidth=0)
    p.draw(window)
    

pauseText = visual.TextStim(window, text='Dear Participant,\n\n' +
                    'You may now take a short break\n' +
                    'Whenever you want to continue:\n\n'+
                    '    Press any key to continue', 
                    antialias=False)
pauseText.wrapWidth=700

readyText = visual.TextStim(window, text='Dear Participant,\n\n' +
                            'Whenever you are ready press a button and the experiment starts.\n\n'+
                            'As a reminder:\n'+
                            'If the two dots fall on the same rectangle, press m,\n'+
                            'If they do not press z.\n\n'+
                            'Only the first press counts and you will see every image for up to 5 seconds.\n\n'+
                            'The two points we ask about will always be the same color.\n'+
                            'We always draw rectangles until the square is filled.\n\n'+
                            '    Press any key to continue', 
                            antialias=False)
readyText.wrapWidth=700


resultsAll = np.zeros((0,4))
# results = [im_size,imageNumber,truth,response]

event.waitKeys()

resList = []
## Main Experiment Script

imageNumber = np.arange(n_trials) + offset_image_number 
results = np.zeros((n_trials,4))*np.nan

results[:,0] = im_size
results[:,1] = imageNumber

# Display Introtext
readyText.draw()
window.flip()
event.waitKeys()

for i in range(len(results)):
    imageNumber = results[i,1]
    resTrial = Trial(window,clock,imageNumber=imageNumber)
    resList.append(resTrial)
    if not (resTrial[0] is None):
      if len(resTrial[0])>0:
        if resTrial[0][0][0]=='z':
            results[i,3] = -1
        elif resTrial[0][0][0]=='m':
            results[i,3] = 1
        elif resTrial[0][0][0]=='q':
            break
        else:
            results[i,3] = 0
    else:
        results[i,3] = 0
        
    if resTrial[1]:
        results[i,2] = 1
    else:
        results[i,2] =-1
        
    print(results[i])
    if (i % 100 == 99):
        correct = np.sum(results[(i-99):(i+1),3] == results[(i-99):(i+1),2])
        pauseText.text= ('Dear Participant,\n\n' +
                'You may now take a short break.\n' +
                'You answered correctly for %d of the last %d images.\n' +
                'Whenever you want to continue:\n\n'+
                '    Press any key to continue') % (correct,100)
        pauseText.draw()
        window.flip()
        event.waitKeys()
resultsAll = np.append(resultsAll,results,axis=0)


now = datetime.datetime.now()
np.save((res_folder+'result_%d_'+ now.strftime('%Y_%m_%d_%H_%M_%S.npy')) % im_size, resultsAll)
with open((res_folder+'result_%d_'+ now.strftime('%Y_%m_%d_%H_%M_%S.pickle')) % im_size, 'wb+') as f:
    pickle.dump(resList,f)
core.wait(3,3)
window.close()

exit()