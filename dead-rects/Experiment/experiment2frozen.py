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

im_folder = './imagesFrozen/'

name = input("What is your subject code?")

res_folder = './resultsFrozen/%s/' % name

n_trials=input('How many trials per condition do you want to do today?')

offset_image_number = input('How far shall I offset the image number?')


border = input('Shall I display borders? [anything is Yes]')
border = bool(border)


if not os.path.exists(im_folder):
    os.mkdir(im_folder)
if not os.path.exists(res_folder):
    os.mkdir(res_folder)

n_trials = int(n_trials) # per condition
offset_image_number = int(offset_image_number)
n_cols = np.array([9])
exponents = np.arange(5)+1
exponents = exponents[np.random.permutation(len(exponents))]

#sizes = 5*np.arange(2,80,dtype='float')
#prob = (sizes/np.min(sizes)) **-1.5
#im_size = np.array([800,800])

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
im_size = np.array([300,300])
factorSize = 3


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
distances = [5,10,20,40,80]
distancesd = [4,7,14,28,57]
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
    




def Trial(window,clock,distance=5,num_colors=9,border=False,angle=0,abs_angle=0,exponent=3,imageNumber=0): 
    im_name = im_folder + 'image%d_%d_%d_%d_%d_%d_%d.png' % (exponent,num_colors,distance,angle,abs_angle,imageNumber,int(border))
    image = PIL.Image.open(im_name)
    rect_name = im_folder + 'rect%d_%d_%d_%d_%d_%d_%d.npy' % (exponent,num_colors,distance,angle,abs_angle,imageNumber,int(border))
    t0 = window.flip()
    CenterImage = visual.ImageStim(window)
    im_pil = image.resize(factorSize*np.array(im_size))
    CenterImage.setImage(im_pil)
    CenterImage.draw(window)
    t1 = window.flip()
    keypresses = event.waitKeys(maxWait=5,timeStamped=clock)
    t2 = window.flip()
    same_name = im_folder + 'same%d_%d_%d_%d_%d_%d_%d.npy' % (exponent,num_colors,distance,angle,abs_angle,imageNumber,int(border))
    same_object = np.load(same_name)
    return im_name,rect_name,keypresses,same_object[0],(t0,t1,t2),same_object[1],same_object[2]

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
    
def run_movie(window,nCol,duration=500,fperRect=1,exponent = 3,border=False): 
    prob = (sizes/np.min(sizes)) ** -(exponent/2)
    dl_mov=dl.dlMovie(im_size=im_size,
          sizes=sizes,
          prob = prob,
          grid=1,
          colors=np.linspace(0,1,nCol),
          border = border)
    stim = visual.ImageStim(window)
    for i in range(duration):
        dl_mov.add_leaf()
        im = np.repeat([dl_mov.image],3,axis=0)
        im[2,im[0]==5]= 1
        im[1,im[0]==5]= 0.5
        im[0,im[0]==5]= 0.5
        im[2,np.isnan(im[0])]= 1
        im[1,np.isnan(im[0])]= 0
        im[0,np.isnan(im[0])]= 0
        im = np.uint8(255*im.transpose((1,2,0)))
        im_pil = PIL.Image.fromarray(im) #.convert('RGB')
        im_pil = im_pil.resize(factorSize*np.array(im_size))
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

pauseText = visual.TextStim(window, text='Dear Participant,\n\n' +
                    'You may now take a short break\n' +
                    'Whenever you want to continue:\n\n'+
                    '    Press any key to continue', 
                    antialias=False)
pauseText.wrapWidth=700

# show movie of dead-leaf generation
movieText = visual.TextStim(window, text='Dear Participant,\n\n' +
                            'Before we start with this set of images, we want to show you \n'+
                            'how the images used in this set are generated.\n'+
                            'We still randomly choose rectangles and place them in random positions.\n'+
                            'To illustrate what rectangles we draw from, we made a movie, which shows this process over time.\n\n'+
                            '    Press any key to start the movie for this block', 
                            antialias=False)
movieText.wrapWidth=700
    
introText = visual.TextStim(window, text='Dear Participant,\n\n' +
                            'you will be shown pictures formed of rectangles.\n' +
                            'Two points will be marked with red markers. Please report, whether you believe they fall on the same rectangle or not.\n\n'+
                            'If they fall on the same rectangle, press m.\n'+
                            'If they do not press z.\n\n'+
                            'The image will be shown for up to 5 seconds.\n\n'+
                            'The two points we ask about will always be the same color.\n\n'+
                            '    Press any key to continue', 
                            antialias=False)
introText.wrapWidth=700



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




resultsAll = np.zeros((0,9))
# results = [exponent,n_colours,distance,angle,abs_angle,imageNumber,truth,response]

introText.draw()
window.flip()

event.waitKeys()

resList = []
## Main Experiment Script
for iBlock in range(len(exponents)):
    exponent = exponents[iBlock] 

    # results = [exponent,n_colours,distance,angle,abs_angle,imageNumber,truth,response,truthReduced]
    imageNumber = np.arange(n_trials) + offset_image_number
    
    results = np.zeros((n_trials*len(n_cols)*(len(distances)+len(distancesd))*2,9))*np.nan
    mesh = np.meshgrid(exponent,n_cols,distances,0,[0,1],imageNumber)
    meshd = np.meshgrid(exponent,n_cols,distancesd,1,[0,1],imageNumber)

    results[:,0] = np.append(mesh[0].flatten(),meshd[0].flatten())
    results[:,1] = np.append(mesh[1].flatten(),meshd[1].flatten())
    results[:,2] = np.append(mesh[2].flatten(),meshd[2].flatten())
    results[:,3] = np.append(mesh[3].flatten(),meshd[3].flatten())
    results[:,4] = np.append(mesh[4].flatten(),meshd[4].flatten())
    results[:,5] = np.append(mesh[5].flatten(),meshd[5].flatten())

    np.random.shuffle(results)
    
    movieText.draw()
    window.flip()
    event.waitKeys()

    run_movie(window,9,border=border,exponent=exponent)

    # Display Introtext
    readyText.draw()
    window.flip()
    event.waitKeys()
    
    for i in range(len(results)):
        exponent = results[i,0]
        n_c = int(results[i,1])
        distance = results[i,2]
        angle = results[i,3]
        abs_angle = results[i,4]
        imageNumber = results[i,5]
        resTrial = Trial(window,clock,num_colors=n_c,distance=distance,border=border,exponent=exponent,angle=angle,abs_angle=abs_angle,imageNumber=imageNumber)
        resList.append(resTrial)
        if not (resTrial[2] is None):
          if len(resTrial[2])>0:
            if resTrial[2][0][0]=='z':
                results[i,7] = -1
            elif resTrial[2][0][0]=='m':
                results[i,7] = 1
            elif resTrial[2][0][0]=='q':
                break
            else:
                results[i,7] = 0
        else:
            results[i,7] = 0
            
        if resTrial[3]:
            results[i,6] = 1
        else:
            results[i,6] =-1
            
        if resTrial[6]:
            results[i,8] = 1
        else:
            results[i,8] =-1
            
        print(results[i])
        if (i % 100 == 99):
            correct = np.sum(results[(i-99):(i+1),7] == results[(i-99):(i+1),6])
            pauseText.text= ('Dear Participant,\n\n' +
                    'You may now take a short break.\n' +
                    'You answered correctly for %d of the last %d images.\n' +
                    'Whenever you want to continue:\n\n'+
                    '    Press any key to continue') % (correct,100)
            pauseText.draw()
            window.flip()
            event.waitKeys()
    if resTrial[2][0][0]=='q':
        break
    resultsAll = np.append(resultsAll,results,axis=0)


now = datetime.datetime.now()
if border:
    np.save(now.strftime(res_folder+'resBorder%Y_%m_%d_%H_%M_%S.npy'),resultsAll)
    with open(now.strftime(res_folder+'resBorderList%Y_%m_%d_%H_%M_%S.pickle'), 'wb+') as f:
        pickle.dump(resList,f)

else:
    np.save(now.strftime(res_folder+'result%Y_%m_%d_%H_%M_%S.npy'),resultsAll)
    with open(now.strftime(res_folder+'resList%Y_%m_%d_%H_%M_%S.pickle'), 'wb+') as f:
        pickle.dump(resList,f)
core.wait(3,3)
window.close()

exit()