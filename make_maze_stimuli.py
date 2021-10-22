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

config = {
    'task_difficulty': 'easy',
    'easy': {
        'n_occluders': 2,
        'occluder_size_mean': 2.5,
        'occluder_size_var': 0.1,
        'pos': ['c', 'c']
    },
    'medium': {
        'n_occluders': 5,
        'occluder_size_mean': 1.5,
        'occluder_size_var': 0.1,
        'pos': ['u', 'd', 'l', 'r', 'c']
    },
    'hard': {
        'n_occluders': 8,
        'occluder_size_mean': 0.8,
        'occluder_size_var': 0.1,
        'pos': ['u', 'd', 'l', 'r', 'c', 'c', 'c', 'c']
    }
}

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

def gen_trial(fake_target_im, path_x, path_y, obj_database, rad, dset_info, res_dir):

    start_im = np.zeros_like(fake_target_im)
    end_im = np.zeros_like(fake_target_im)
    cv2.circle(start_im, (path_x[0], path_y[0]), 2*rad, (255, 255, 255), -1)    
    cv2.circle(end_im, (path_x[-1], path_y[-1]), 2*rad, (255, 255, 255), -1)    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad + 1, 2*rad + 1))
    finished = False
    stim_num = 1

    # Determine the trial type [0 (negative) / 1 (positive)]
    trial_choice = np.random.choice(2)
    is_neg_trial = True
    trial_type = 'neg'

    if trial_choice == 1:
        is_neg_trial = False
        trial_type = 'pos'

    while not finished:

        target_im = np.zeros_like(fake_target_im)
        
        # randomly pick occluders from the database, as per task difficulty
        n_objs = config[config['task_difficulty']]['n_occluders']
        obj_size = config[config['task_difficulty']]['occluder_size_mean']
        obj_size_var = config[config['task_difficulty']]['occluder_size_var']

        shuff_objs = np.random.permutation(len(obj_database))[:n_objs]
        order = config[config['task_difficulty']]['pos']

        for idx, obj in enumerate(shuff_objs):
            im = obj_database[obj]
            condition_satisfied = False
            c_iter = 0       
            while not condition_satisfied and c_iter < 30:
                c_iter += 1

                # (pos, rot, scale)
                p,q = np.random.randint(low=0, high=224), np.random.randint(low=0, high=224)
                rot = np.random.randint(low=0, high=360)
                scale = np.random.normal(loc=obj_size, scale=obj_size_var)

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
                if (not np.logical_and(target_im_copy, start_im).any()) and (not np.logical_and(target_im_copy, end_im).any()):
                    target_im[x1:x2, y1:y2] = rot_im[:(x2-x1), :(y2-y1)] 
                    condition_satisfied = True
        
            if condition_satisfied != True:
                return False

        outim, retval = check_path_exists(target_im, kernel, path_x, path_y)

        if False:
            plt.subplot(121); 
            plt.imshow(outim); 
            plt.subplot(122); 
            plt.imshow(target_im);
            plt.imshow(start_im, alpha=0.25);
            plt.imshow(end_im, alpha=0.25); 
            plt.show();

            import ipdb; ipdb.set_trace()
        finished = (is_neg_trial == (not retval))

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(target_im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # draw all contours
    color_im = np.tile(target_im[:,:,np.newaxis],(1,1,3)).astype(np.uint8)
    color_im = cv2.drawContours(color_im, contours, -1, (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(color_im, (path_x[-1], path_y[-1]), rad, (255, 255, 255), 2)    
   
    # draw the circle
    cv2.circle(color_im, (path_x[0], path_y[0]), rad, (255, 255, 255), -1)

    im_idx = dset_info[trial_type]
    cv2.imwrite(os.path.join(res_dir, trial_type, 'stim_image_%s_%06d.png'%(trial_type,im_idx)), color_im)
    dset_info[trial_type] += 1

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

        #import ipdb; ipdb.set_trace()
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

        if (n_objects == 2) and components.shape[0] > 1:        
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


def gen_maze_stim(obj_database, dset_info, res_dir):
    # create the canvas
    h,w = 224, 224

    # generate a random path 
    trial_complete = False
    while not trial_complete:
        # create a ball of random radius
        rad = np.random.randint(low=10, high=30)

        # create a path (must control for length)
        start_x = np.random.randint(low=0, high=224)
        start_y = np.random.randint(low=0, high=224)
        length = np.random.randint(low=5, high=10)

        path_x, path_y = gen_path(start_x, start_y, radius=rad, length=length)
        X = np.logical_and(path_x < (224-rad), path_x > rad)
        Y = np.logical_and(path_y < (224-rad), path_y > rad)

        if X.all() and Y.all():
            target_im = np.zeros((h,w))
            fake_target_im = np.zeros((h,w))

            for k in range(len(path_x)):
                cv2.circle(fake_target_im, (path_x[k], path_y[k]), rad, (255, 255, 255), -1)

            trial_complete = gen_trial(fake_target_im, path_x, path_y, obj_database, rad, dset_info, res_dir)
         
def main():
    obj_database = make_obj_database() 
    res_dir = '/media/data_cifs/projects/prj_neural_circuits/maze'
    dset_info = {'pos':0, 'neg':0}

    for k in tqdm.tqdm(range(200000)):
        gen_maze_stim(obj_database, dset_info, res_dir)


if __name__ == '__main__':
    main()
