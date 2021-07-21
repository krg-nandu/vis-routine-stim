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

# remove examples close to the boundary
# positive samples in two different islands
# introduce clutter
# introduce difficulty levels
def gen_stim(obj_database, 
             idx,
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

    # get the position of the X mark
    cross_x, cross_y = np.random.randint(low=10,high=h-10), np.random.randint(low=10, high=w-10)

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(target_im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = [x for x in contours if x.shape[0] > 10]    

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

    label = (color_im2[cross_y, cross_x] == 255)
    label = label.any()    
    prefix = 'neg'
    if label:
        prefix = 'pos'

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
    plt.savefig('/media/data_cifs/projects/prj_neural_circuits/insideness/%s/stim_image_%06d.png'%(prefix, idx), dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
 
    '''
    dpi = 96
    fig = Figure((224/dpi, 224/dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.imshow(color_im)
    for contour in contours:
        xlist = list(contour[:,0,0])
        ylist = list(contour[:,0,1])
        ax.plot(xlist, ylist, '--', linewidth=contour_thickness, dashes=(contour_length, contour_sep), color=[1,1,1])
    
    ax.scatter(cross_x, cross_y, 25, 'w')
    ax.scatter(cX, cY, 25, 'w')
    ax.axis('off')
    plt.tight_layout()
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(224,224,3)

    prefix = 'neg'
    if label:
        prefix = 'pos'

    plt.show()
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(color_im, 'X', (cross_y, cross_x), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    #cv2.circle(color_im, (cross_y, cross_x), 5, (255,255,255), -1)
    cv2.imwrite('/media/data_cifs/projects/prj_neural_circuits/insideness/%s/stim_image_%06d.png'%(prefix, idx), image)
    #plt.savefig('/media/data_cifs/projects/prj_neural_circuits/insideness/%s/stim_image_%06d.png'%(prefix, idx))
    plt.close()
    '''

def gen_path(sx,sy, radius=5, length=14):
    path_x, path_y = [sx], [sy]
    # lets treat the arclength as radius now
    for seg in range(length):
        theta = (np.random.randint(low=-90,high=90.)/180.)*np.pi
        for k in range(5):
            newx = int(path_x[-1] + radius*np.cos(theta))
            newy = int(path_y[-1] + radius*np.sin(theta))
            path_x.append(newx)
            path_y.append(newy)
    return np.array(path_x), np.array(path_y)

def gen_positive_trial(fake_target_im, path_x, path_y, obj_database, rad, im_idx, res_dir):
    target_im = np.zeros_like(fake_target_im)

    # randomly pick 5 objects from the object database
    shuff_objs = np.random.permutation(len(obj_database))[:7]
    order = ['u', 'd', 'l', 'r', 'c', 'c', 'c']
    for idx, obj in enumerate(shuff_objs):
        im = obj_database[obj]
        condition_satisfied = False        
        while not condition_satisfied:
            # (pos, rot, scale)
            p,q = np.random.randint(low=0, high=224), np.random.randint(low=0, high=224)
            rot = np.random.randint(low=0, high=360)
            scale = 0.5 + np.random.rand()*1.
            rot_im = imutils.rotate_bound(im.astype(np.uint8), rot)
            rot_im = cv2.resize(rot_im, (int(rot_im.shape[0]*scale), int(rot_im.shape[1]*scale)))
            dims = rot_im.shape
 
            if order[idx] == 'u':
                rot_im = rot_im[int(np.floor(dims[0]/2)):, :]
                loc = (int(np.floor(dims[0]/4)),q)
            elif order[idx] == 'd':
                rot_im = rot_im[:int(np.floor(dims[0]/2)), :]
                loc = (224-int(np.floor(dims[0]/4)),q)
            elif order[idx] == 'l':
                loc = (p,0)
                rot_im = rot_im[:, int(np.floor(dims[1]/2)):]
                loc = (p, int(np.floor(dims[1]/4)))
            elif order[idx] == 'r':
                rot_im = rot_im[:, :int(np.floor(dims[1]/2))]
                loc = (p, 224-int(np.floor(dims[1]/4)))
            else:
                loc = (p,q)
            
            # need to check if this satisfies
            sh = [int(np.floor(rot_im.shape[0]/2)), int(np.floor(rot_im.shape[1]/2))]
            target_im_copy = target_im.copy()
            x1 = max(loc[0]-sh[0],0)
            x2 = min(loc[0]+sh[0],224)

            y1 = max(loc[1]-sh[1],0)
            y2 = min(loc[1]+sh[1],224)

            target_im_copy[x1:x2, y1:y2] = rot_im[:(x2-x1), :(y2-y1)]   
            
            if not np.logical_and(target_im_copy, fake_target_im).any():
                target_im[x1:x2, y1:y2] = rot_im[:(x2-x1), :(y2-y1)] 
                condition_satisfied = True

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(target_im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = [x for x in contours if x.shape[0] > 10]    

    color_im = np.tile(target_im[:,:,np.newaxis],(1,1,3)).astype(np.uint8)
    # draw all contours
    color_im = cv2.drawContours(color_im, contours, -1, (255, 255, 255), 2)

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(color_im, 'X', (path_x[-1], path_y[-1]), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(color_im, (path_x[-1], path_y[-1]), rad, (255, 255, 255), 2)    

    # draw the circle
    cv2.circle(color_im, (path_x[0], path_y[0]), rad, (255, 255, 255), -1)    
    #plt.imshow(color_im)
    #plt.show()
    cv2.imwrite(os.path.join(res_dir, 'pos', 'stim_image_pos_%06d.png'%im_idx), color_im)


def check_path_exists(target_im, kernel, path_y, path_x):
    total_px = kernel.sum()
    dims = target_im.shape
    
    # invert the target image and then convolve
    inv_im = cv2.bitwise_not((target_im * 255.).astype(np.uint8))/255.
    outim = S.convolve(inv_im, kernel, mode='constant', cval=0.)

    # at location (x,y) the value is True if the disc can legally pass through (x,y)
    p_graph = (outim == total_px)

    G = nx.empty_graph()

    for r in range(1, dims[0]-1):
        for c in range(1, dims[1]-1):
            n_id = r*dims[0] + c
            G.add_node(n_id)

            neighbors = [(r-1,c-1), (r-1,c), (r-1,c+1), (r,c-1), (r,c+1), (r+1,c-1), (r+1, c), (r+1, c+1)]
            node_ids = [x*dims[0] + y for (x,y) in neighbors]

            if p_graph[r,c]:
                for node,node_id in zip(neighbors, node_ids):
                    if p_graph[node[0], node[1]]:   
                        G.add_edge(n_id, node_id)

    start_node = path_x[0] * dims[0] + path_y[0]
    end_node = path_x[-1] * dims[0] + path_y[-1]

    return outim, nx.has_path(G, start_node, end_node)

def gen_negative_trial(fake_target_im, path_x, path_y, obj_database, rad, im_idx, res_dir):

    start_im = np.zeros_like(fake_target_im)
    end_im = np.zeros_like(fake_target_im)
    cv2.circle(start_im, (path_x[0], path_y[0]), rad, (255, 255, 255), -1)    
    cv2.circle(end_im, (path_x[-1], path_y[-1]), rad, (255, 255, 255), -1)    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad,2*rad))
    finished = False
    stim_num = 1

    while not finished:
        print(stim_num)
        stim_num+=1

        target_im = np.zeros_like(fake_target_im)
        
        # randomly pick 7 objects from the object database
        shuff_objs = np.random.permutation(len(obj_database))[:7]
        order = ['u', 'd', 'l', 'r', 'c', 'c', 'c']
        for idx, obj in enumerate(shuff_objs):
            im = obj_database[obj]
            condition_satisfied = False
            c_iter = 0       
            while not condition_satisfied and c_iter < 10:
                c_iter += 1

                # (pos, rot, scale)
                p,q = np.random.randint(low=0, high=224), np.random.randint(low=0, high=224)
                rot = np.random.randint(low=0, high=360)
                scale = 0.5 + np.random.rand()*1.
                rot_im = imutils.rotate_bound(im.astype(np.uint8), rot)
                rot_im = cv2.resize(rot_im, (int(rot_im.shape[0]*scale), int(rot_im.shape[1]*scale)))
                dims = rot_im.shape
     
                if order[idx] == 'u':
                    rot_im = rot_im[int(np.floor(dims[0]/2)):, :]
                    loc = (int(np.floor(dims[0]/4)),q)
                elif order[idx] == 'd':
                    rot_im = rot_im[:int(np.floor(dims[0]/2)), :]
                    loc = (223-int(np.floor(dims[0]/4)),q)
                elif order[idx] == 'l':
                    loc = (p,0)
                    rot_im = rot_im[:, int(np.floor(dims[1]/2)):]
                    loc = (p, int(np.floor(dims[1]/4)))
                elif order[idx] == 'r':
                    rot_im = rot_im[:, :int(np.floor(dims[1]/2))]
                    loc = (p, 223-int(np.floor(dims[1]/4)))
                else:
                    loc = (p,q)
                
                # need to check if this satisfies
                sh = [int(np.floor(rot_im.shape[0]/2)), int(np.floor(rot_im.shape[1]/2))]
                target_im_copy = copy.deepcopy(target_im)
                x1 = max(loc[0]-sh[0],0)
                x2 = min(loc[0]+sh[0],224)

                y1 = max(loc[1]-sh[1],0)
                y2 = min(loc[1]+sh[1],224)

                target_im_copy[x1:x2, y1:y2] = rot_im[:(x2-x1), :(y2-y1)]   
                #import ipdb; ipdb.set_trace()            
                if np.logical_and(target_im_copy, fake_target_im).any() and (not np.logical_and(target_im_copy, start_im).any()) and (not np.logical_and(target_im_copy, end_im).any()):
                    target_im[x1:x2, y1:y2] = rot_im[:(x2-x1), :(y2-y1)] 
                    condition_satisfied = True
        
            if condition_satisfied != True:
                return

        outim, retval = check_path_exists(target_im, kernel, path_x, path_y)
        finished = not retval

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(target_im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    color_im = np.tile(target_im[:,:,np.newaxis],(1,1,3)).astype(np.uint8)
    # draw all contours
    color_im = cv2.drawContours(color_im, contours, -1, (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(color_im, 'X', (path_x[-1], path_y[-1]), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(color_im, (path_x[-1], path_y[-1]), rad, (255, 255, 255), 2)    
   
    # draw the circle
    cv2.circle(color_im, (path_x[0], path_y[0]), rad, (255, 255, 255), -1)    
    #plt.subplot(121); plt.imshow(outim); plt.subplot(122); plt.imshow(color_im)
    #plt.show()

    cv2.imwrite(os.path.join(res_dir, 'neg', 'stim_image_neg_%06d.png'%im_idx), color_im)


def make_obj_database():
    obj_database = {}
    obj_id = 0
    kernel = np.ones((25,25), np.float32)/625

    # make a database of objects
    for idx in tqdm.tqdm(range(200)):
        mask = cv2.imread('Desktop/synthetic_objects/image_%06d.png'%idx)
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

        if n_objects == 2:        
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


def gen_maze_stim(obj_database, im_idx, res_dir):
    # create the canvas
    h,w = 224, 224
    target_im = np.zeros((h,w))
    fake_target_im = np.zeros((h,w))

    # get the path 
    while True:
        # create a ball of random radius
        rad = np.random.randint(low=10, high=20)

        # create a path (must control for length)
        start_x = np.random.randint(low=0, high=224)
        start_y = np.random.randint(low=0, high=224)
        #length = np.random.randint(low=5, high=10)
        length = 8
        path_x, path_y = gen_path(start_x, start_y, radius=rad, length=length)
        X = np.logical_and(path_x < (224-rad), path_x > rad)
        Y = np.logical_and(path_y < (224-rad), path_y > rad)
        if X.all() and Y.all():
            break

    # make a "fake" target image
    for k in range(len(path_x)):
        cv2.circle(fake_target_im, (path_x[k], path_y[k]), rad, (255, 255, 255), -1)

    #plt.imshow(fake_target_im)
    #plt.show(block=False)

    # 0: negative, 1: positive
    trial_type = np.random.choice(2)
    gen_positive_trial(fake_target_im, path_x, path_y, obj_database, rad, im_idx, res_dir)
    gen_negative_trial(fake_target_im, path_x, path_y, obj_database, rad, im_idx, res_dir)
   
def main():
    obj_database = make_obj_database() 
    res_dir = '/media/data_cifs/projects/prj_neural_circuits/maze'
    for k in tqdm.tqdm(range(100)):
        gen_maze_stim(obj_database, k, res_dir)

    #for k in tqdm.tqdm(range(100)):
    #    gen_stim(obj_database, k)

if __name__ == '__main__':
    main()
