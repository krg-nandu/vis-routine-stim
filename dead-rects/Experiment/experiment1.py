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

# This is the only way I found to do this. It looks like a HORRIBLE IDEA
# but unfortunately it works so far.
import sys
sys.path.append('..')
import DeadLeaf as dl
sys.path.pop(-1)

import datetime

monitors.monitorFolder= '/Users/heiko/.psychopy3/monitors'

im_folder = './images/'

name = input("What is your subject code?")

res_folder = './results/%s/' % name

n_trials=input('How many trials per condition do you want to do today?')

if not os.path.exists(im_folder):
    os.mkdir(im_folder)
if not os.path.exists(res_folder):
    os.mkdir(res_folder)

n_trials = int(n_trials) # per condition
n_cols = [2,3,4,6,8]

#sizes = 5*np.arange(2,80,dtype='float')
#prob = (sizes/np.min(sizes)) **-1.5
#imSize = np.array([800,800])

#Sizes chosen for horizontal:
#3 0.8972123464097765
#8 0.7475256304569095
#24 0.4925513111748433
#68 0.2491748800639651
#146 0.09892203196770742
#Sizes chosen for diagonal:
#2 0.8675302830216124
#5 0.7058433239505979
#11 0.48098719795795203
#28 0.24386559658864032
#62 0.09915052609944638
#distances = [3,8,24,68,146]
#distancesd = [2,5,11,28,62]

sizes = 5*np.arange(1,80,dtype='float')
prob = (sizes/np.min(sizes)) **-1.5
imSize = np.array([400,400])
factorSize = 2


#Sizes chosen for horizontal:
#2 0.8897441130089403
#5 0.7454126020552115
#19 0.49815449458047945
#62 0.24906879663087056
#140 0.09888733852162014
#Sizes chosen for diagonal:
#1 0.8912648809369313
#3 0.7132471428644057
#8 0.47627923219339785
#23 0.2460755676513367
#56 0.0992763294998463
distances = [5,19,62,140]
distancesd = [3,8,23,56]
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
    




def Trial(window,clock,num_colors=8,positions=[]):
    positions = np.floor(positions)
    positions_im = np.zeros_like(positions)
    positions_im[:,1] = np.floor(imSize/2)+positions[:,0]
    positions_im[:,0] = np.floor(imSize/2)-positions[:,1]-1
    col = np.random.randint(num_colors)
    im = dl.gen_rect_leaf(imSize,
          sizes=sizes,
          prob = prob,
          grid=1,
          colors=np.linspace(0,1,num_colors),
          fixedIdx = positions_im,
          fixedC=col)
    now = datetime.datetime.now()
    im_name = now.strftime(im_folder+"image%Y_%m_%d_%H_%M_%S_%f.png")
    im_pil = PIL.Image.fromarray(255*im[0]).convert('RGB')
    im_pil.save(im_name,'PNG')
    rect_name = now.strftime(im_folder+"rect%Y_%m_%d_%H_%M_%S_%f.npy")
    np.save(rect_name,im[1])
    same_object = False
    t0 = window.flip()
    CenterImage = visual.ImageStim(window)
    im_pil = im_pil.resize(factorSize*np.array(imSize))
    CenterImage.setImage(im_pil)
    CenterImage.draw(window)
#    if len(positions)>0 and angle:
#        draw_pos_marker(window,positions[0])
#        draw_pos_marker(window,positions[1])
#    elif len(positions)>0 and not angle:
#        draw_pos_marker_diagonal(window,positions[0])
#        draw_pos_marker_diagonal(window,positions[1])
    draw_pos_marker_dot(window,factorSize*positions[0])
    draw_pos_marker_dot(window,factorSize*positions[1])
    t1 = window.flip()
    #core.wait(5,5)
    keypresses = event.waitKeys(maxWait=5,timeStamped=clock)
    t2 = window.flip()
    #keypresses = event.getKeys(None,False,clock)
    same_object = im[2]
    return im_name,rect_name,keypresses,same_object,(t0,t1,t2),positions,positions_im,col

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
    
def run_movie(window,nCol,duration=1500,fperRect=1):
    dl_mov=dl.dlMovie(imSize=imSize,
          sizes=sizes,
          prob = prob,
          grid=1,
          colors=np.linspace(0,1,nCol))
    stim = visual.ImageStim(window)
    for i in range(duration):
        dl_mov.add_leaf()
        im = np.repeat([dl_mov.image],3,axis=0)
        im[2,np.isnan(im[0])]= 1
        im[1,np.isnan(im[0])]= 0
        im[0,np.isnan(im[0])]= 0
        im = np.uint8(255*im.transpose((1,2,0)))
        im_pil = PIL.Image.fromarray(im) #.convert('RGB')
        im_pil = im_pil.resize(factorSize*np.array(imSize))
        stim.setImage(im_pil)
        stim.draw(window)
        for j in range(fperRect-1):
            window.flip(clearBuffer=False)
        window.flip()
#im = dl.gen_rect_leaf([1,1],
#          sizes=sizes,
#          prob = prob,
#          grid=1,
#          colors=np.linspace(0,1,3))
#im = [np.array([[1.0, 0] * 3] * 20).transpose()]
#
#im_pil = PIL.Image.fromarray(255*im[0]).convert('RGB')
#CenterImage = visual.ImageStim(window)
#CenterImage.setImage(im_pil)
#CenterImage.draw(window)
#draw_pos_marker(window,np.array([0,0]))
#draw_pos_marker(window,np.array([0,3]))
#window.flip()
#core.wait(500,500)    



## Main Experiment Script
# results = [n_colours,distance,angle,abs_angle,truth,response]
results = np.zeros((n_trials*len(n_cols)*(len(distances)+len(distancesd)),6))*np.nan
mesh = np.meshgrid(n_cols,distances,0)
meshd = np.meshgrid(n_cols,distancesd,1)

results[:,0] = np.append(np.repeat(mesh[0].flatten(),n_trials),np.repeat(meshd[0].flatten(),n_trials))
results[:,1] = np.append(np.repeat(mesh[1].flatten(),n_trials),np.repeat(meshd[1].flatten(),n_trials))
results[:,2] = np.append(np.repeat(mesh[2].flatten(),n_trials),np.repeat(meshd[2].flatten(),n_trials))
results[:,3] = np.random.randint(2,None,(n_trials*len(n_cols)*(len(distances)+len(distancesd))))

np.random.shuffle(results)
resList = []



pauseText = visual.TextStim(window, text='Dear Participant,\n\n' +
                    'You may now take a break\n' +
                    'Whenever you want to continue:\n\n'+
                    '    Press any key to continue', 
                    antialias=False)
pauseText.wrapWidth=700

# show movie of dead-leaf generation
introText = visual.TextStim(window, text='Dear Participant,\n\n' +
                            'Before we start with the main experiment we want to show you \n'+
                            'how the images used in the experiment are generated.\n'+
                            'We randomly choose rectangles and place them in random positions.\n'+
                            'To illustrate this, we made a movie, which shows this process over time.\n\n'+
                            '    Press any key to start the movie', 
                            antialias=False)
introText.wrapWidth=700
introText.draw()
window.flip()

event.waitKeys()

run_movie(window,10)


# Display Introtext
introText = visual.TextStim(window, text='Dear Participant,\n\n' +
                            'you will be shown pictures formed of rectangles as you just saw.\n' +
                            'Two points will be marked with red markers. Please report, whether you believe they fall on the same rectangle or not.\n\n'+
                            'If they fall on the same rectangle, press m,\n'+
                            'if they do not press z.\n\n'+
                            'Only the first press counts and you will get 5 seconds for each image.\n\n'+
                            'The number of different colours will vary.\n'+
                            'However the two points we ask about will always be the same color.\n\n'+
                            '    Press any key to continue', 
                            antialias=False)
introText.wrapWidth=700
introText.draw()
window.flip()

event.waitKeys()

for i in range(len(results)):
    distance = results[i,1]
    n_c = int(results[i,0])
    angle = results[i,2]
    abs_angle = results[i,3]
    if angle and not abs_angle:
        pos = [[-distance/2,-distance/2],[distance/2,distance/2]]
    elif angle and abs_angle:
        pos = [[-distance/2,distance/2],[distance/2,-distance/2]]
    elif not angle and not abs_angle:
        pos = [[-distance/2,0],[distance/2,0]]
    elif not angle and abs_angle: 
        pos = [[0,-distance/2],[0,distance/2]]
    pos = np.floor(np.array(pos))
    resList.append(Trial(window,clock,num_colors=n_c,positions=pos))
    if not (resList[i][2] is None):
      if len(resList[i][2])>0:
        if resList[i][2][0][0]=='z':
            results[i,5] = -1
        elif resList[i][2][0][0]=='m':
            results[i,5] = 1
        elif resList[i][2][0][0]=='q':
            break
        else:
            results[i,5] = 0
    else:
        results[i,5] = 0
    if resList[i][3]:
        results[i,4] = 1
    else:
        results[i,4] =-1
    print(results[i])
    if (i % 100 == 99):
        pauseText.draw()
        window.flip()
        event.waitKeys()



now = datetime.datetime.now()
np.save(now.strftime(res_folder+'result%Y_%m_%d_%H_%M_%S.npy'),results)
with open(now.strftime(res_folder+'resList%Y_%m_%d_%H_%M_%S.pickle'), 'wb+') as f:
    pickle.dump(resList,f)
core.wait(3,3)
window.close()

exit()