import numpy as np
import OpenEXR
import matplotlib.pyplot as plt
import os, cv2, imutils
import tqdm
import scipy.ndimage as S
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import copy
import networkx as nx

def pick_top_k(nlabels, label_im, k):
    component_sizes = [np.where(label_im == i)[0].shape[0] for i in range(1,nlabels)]
    return np.flip(np.argsort(component_sizes)[-k:] + 1)

def regularize(x, y):
    M = 1000
    t = np.linspace(0, len(x), M)
    x = np.interp(t, np.arange(len(x)), x)
    y = np.interp(t, np.arange(len(y)), y)
    tol = 1.5
    i, idx = 0, [0]
    while i < len(x):
        total_dist = 0
        for j in range(i+1, len(x)):
            total_dist += math.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2)
            if total_dist > tol:
                idx.append(j)
                break
        i = j+1

    xn = x[idx]
    yn = y[idx]
    return xn, yn

# introduce clutter
# introduce difficulty levels
def gen_stim(obj_database, 
             idx,
             res_dir,
             contour_thickness=1,
             contour_length=5,
             contour_sep=5):

    # create the canvas
    h,w = 224, 224
    target_im = np.zeros((h,w))
    n_objects = len(obj_database)    
    idxrand = np.random.randint(n_objects,size=2) 
    
    # extract the first object
    im1 = obj_database[idxrand[0]]
    
    rot = np.random.randint(low=0, high=360)
    scale = 0.5 + np.random.rand()*2
    im1 = imutils.rotate_bound(im1.astype(np.uint8), rot)
    im1 = cv2.resize(im1, (int(im1.shape[0]*scale), int(im1.shape[1]*scale)))
 
    '''
    coords = np.where(label_im == components[1])    
    xmin, xmax, ymin, ymax = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
    im1 = label_im[xmin:xmax, ymin:ymax] == components[1]
    '''
    im1_dims = im1.shape

    # extract the second object
    im2 = obj_database[idxrand[1]]
    rot = np.random.randint(low=0, high=360)
    scale = 0.5 + np.random.rand()*2
    im2 = imutils.rotate_bound(im2.astype(np.uint8), rot)
    im2 = cv2.resize(im2, (int(im2.shape[0]*scale), int(im2.shape[1]*scale)))
 
    '''
    coords = np.where(label_im == components[0])    
    xmin, xmax, ymin, ymax = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
    im2 = label_im[xmin:xmax, ymin:ymax] == components[0]
    '''
    im2_dims = im2.shape

    # pick random offsets for the two objects
    lH, lW = max(im1.shape[0], im2.shape[0]), max(im1.shape[1], im2.shape[1]) 
    rx1, ry1 = np.random.randint(max(1, h - lH)), np.random.randint(max(1, w - lW))

    # draw on the canvas
    offx, offy = np.random.randint(0, 10), np.random.randint(0, 10)

    ulx = min(0, target_im.shape[0] - rx1 - im1.shape[0])
    uly = min(0, target_im.shape[1] - ry1 - im1.shape[1])
    target_im[rx1:rx1+im1.shape[0], ry1:ry1+im1.shape[1]] = im1[:(im1_dims[0]+ulx), :(im1_dims[1]+uly)]

    ulx = min(0, target_im.shape[0] - offx - rx1 - im2.shape[0])
    uly = min(0, target_im.shape[1] - offy - ry1 - im2.shape[1]) 
    target_im[rx1+offx:rx1+offx+im2.shape[0], ry1+offy:ry1+offy+im2.shape[1]] = np.logical_or(target_im[rx1+offx:rx1+offx+im2.shape[0], ry1+offy:ry1+offy+im2.shape[1]], im2[:(im2_dims[0]+ulx), :(im2_dims[1]+uly)])

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(target_im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = [x for x in contours if x.shape[0] > 10]    

    if len(contours) > 1:
        return False

    color_im = np.tile(target_im[:,:,np.newaxis],(1,1,3)).astype(np.uint8)
    color_im2 = np.zeros_like(color_im)
    
    # draw all contours
    #target_im = cv2.drawContours(color_im, contours, -1, (255, 255, 255), 2)
    func = S.filters.gaussian_filter1d

    ZZ = np.vstack(contours).squeeze()
    cv2.fillPoly(color_im2, pts=[ZZ], color=(255,255,255))

    M = cv2.moments(contours[0])         
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # get the position of the X mark
    label = None
    cross_x, cross_y = 0, 0
    while label == None:
        cross_x, cross_y = np.random.randint(low=10,high=h-10), np.random.randint(low=10, high=w-10)
        patch = color_im2[cross_y-10:cross_y+10, cross_x-10:cross_x+10]
        if patch.all():
            # well inside the object
            label = 'pos'
        elif (~patch).all():
            # well outside the object
            label = 'neg'
    prefix = label

    #label = (color_im2[cross_y, cross_x] == 255)
    #label = label.any()    
    #prefix = 'neg'
    #if label:
    #    prefix = 'pos'

    dpi=96
    fig = plt.figure(figsize=(224/dpi, 224/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(color_im)

    for contour in contours:
        xlist = list(contour[:,0,0])
        ylist = list(contour[:,0,1])
        ax.plot(xlist, ylist, '--', linewidth=contour_thickness, dashes=(contour_length, contour_sep), color=[1,1,1])
    
    ax.scatter(cross_x, cross_y, 25, 'w')
    ax.scatter(cX, cY, 25, 'w')
    ax.axis('tight')
    ax.axis('off') 
    plt.savefig(os.path.join(res_dir, '%s/stim_image_%06d.png'%(prefix, idx)), dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    return True

def make_obj_database():
    obj_database = {}
    obj_id = 0
    kernel = np.ones((25,25), np.float32)/625

    # make a database of objects
    for idx in tqdm.tqdm(range(400)):
        mask = cv2.imread('/home/lakshmi/Desktop/synthetic_objects/image_%06d.png'%idx)
        mask_gray = mask[:,:,0]

        # get the connected components
        nlabels, label_im = cv2.connectedComponents(mask_gray)
        # let us assume we pick the first two for now
        n_objects = min(2,nlabels)

        if n_objects == 0:
            continue

        components = pick_top_k(nlabels, label_im, n_objects)
        # extract the first object
        coords = np.where(label_im == components[0])
        if coords[0].shape[0] < 50:
            continue    
        xmin, xmax, ymin, ymax = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
        im1 = label_im[xmin:xmax, ymin:ymax] == components[0]
        im1 = cv2.filter2D(cv2.resize(im1.astype(np.uint8), (100,100)), -1, kernel)
        im1_dims = im1.shape
        obj_database.update({obj_id:im1})
        obj_id += 1

        if (n_objects == 2) and (components.shape[0] > 1):        
            # extract the second object
            coords = np.where(label_im == components[1])    
            if coords[0].shape[0] < 50:
                continue    
            xmin, xmax, ymin, ymax = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
            im2 = label_im[xmin:xmax, ymin:ymax] == components[1]
            im2 = cv2.filter2D(cv2.resize(im2.astype(np.uint8), (100,100)), -1, kernel)
            im2_dims = im2.shape

            obj_database.update({obj_id:im2})
            obj_id += 1
    return obj_database

def main():
    obj_database = make_obj_database() 
    res_dir = '/media/data_cifs/projects/prj_neural_circuits/insideness'
    N_SAMPLES = 1e5
    k = 0

    while k < 100:
        if k%1000 == 0:
            print('Generated {} samples'.format(k))
        ret = gen_stim(obj_database, k, res_dir)
        if ret:
            k += 1

if __name__ == '__main__':
    main()
