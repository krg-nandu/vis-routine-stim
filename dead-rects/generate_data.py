#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable
import os
import numpy as np
import tqdm
import PIL
import json
import DeadLeaf as dl


def main(distances=None, size=100, N=100,
         n_colors=9, alphas=3, max_size=None,
         folder='./data_test', same_color=0, split_folders=0):
    os.makedirs(folder, exist_ok=True)
    if max_size is None:
        max_size = size
    im_size = [size, size]
    sizes = np.arange(5, max_size, 5)
    colors = np.linspace(0, 1, n_colors + 1)[:-1]
    file_name = folder + '.csv'
    file = open(file_name, 'w')
    file.write('idx,fname,x1,y1,x2,y2,distance,d1,d2,col,col2,alpha,n_colors,same\n')
    n_digits = np.ceil(np.log10(N))
    if distances is None:
        dist_probabilities = np.zeros(im_size)
        dist_probabilities[:int(size / 2), :int(size / 2)] = 1
    elif isinstance(distances, str):
        if os.path.isfile(distances):
            dist_probabilities = np.load(distances)
        else:
            distances = json.loads(distances)
    if isinstance(distances, Iterable):
        distances = np.array(distances)
        if distances.shape[1] == 2:
            dist_probabilities = np.zeros(im_size)
            for i_d in distances:
                dist_probabilities[i_d[0], i_d[1]] = 1
        else:
            dist_probabilities = distances
    assert dist_probabilities.shape[0] == im_size[0], 'wrong distance probability size given'
    assert dist_probabilities.shape[1] == im_size[1], 'wrong distance probability size given'
    dist_probabilities = (dist_probabilities / np.sum(dist_probabilities)).flatten()
    for i in tqdm.trange(N):
        if isinstance(alphas, Iterable):
            alpha = alphas[np.random.randint(len(alphas))]
        else:
            alpha = alphas
        r = np.random.rand()
        idx = np.where(np.cumsum(dist_probabilities) > r)[0][0]
        dx, dy = np.unravel_index(idx, im_size)
        select_size = im_size - np.array((dx, dy))
        x = np.random.randint(select_size[0])
        y = np.random.randint(select_size[1])
        if np.random.rand() > 0.5:
            fixedIdx = np.array([[x, y], [x + dx, y + dy]])
        else:
            fixedIdx = np.array([[x + dx, y], [x, y + dy]])
        if split_folders == 0:
            fname = f'%0{n_digits}d' % i
        else:
            k = int(np.floor(i / 10 ** split_folders))
            l = i % (10 ** split_folders)
            f_digits = n_digits - split_folders
            fname = f'%0{f_digits}d/%0{split_folders}d' % (k, l)
            os.makedirs(os.path.join(folder, f'%0{f_digits}d' % k), exist_ok=True)
        fname = fname + '.png'
        prob = (sizes / np.min(sizes)) ** -(alpha / 2)
        if same_color == 0:
            im, rect, same = dl.gen_rect_leaf(
                [size, size], sizes=sizes, prob=prob, colors=colors)
        elif same_color == 1:
            col_i = np.random.randint(len(colors))
            im, rect, same = dl.gen_rect_leaf(
                im_size, sizes=sizes, prob=prob, colors=colors,
                fixedC=col_i, fixedIdx=fixedIdx)
        elif same_color == 2:
            repeat = True
            while repeat:
                im, rect, same = dl.gen_rect_leaf(
                    [size, size], sizes=sizes, prob=prob, colors=colors)
                if (im[fixedIdx[0][0], fixedIdx[0][1]] ==
                    im[fixedIdx[1][0], fixedIdx[1][1]]):
                    repeat = False
        col = im[fixedIdx[0][0], fixedIdx[0][1]]
        col2 = im[fixedIdx[1][0], fixedIdx[1][1]]
        same = dl.test_positions(rect, fixedIdx)
        distance = np.sqrt(dx ** 2 + dy ** 2)
        file.write(f'{i},{fname},{fixedIdx[0][0]},{fixedIdx[0][1]},' +
                   f'{fixedIdx[1][0]},{fixedIdx[1][1]},{distance},' +
                   f'{dx},{dy},{col},{col2},{alpha},{n_colors},{same}\n')
        im[fixedIdx[0][0], fixedIdx[0][1]] = 1
        im[fixedIdx[1][0], fixedIdx[1][1]] = 1
        pil_i = PIL.Image.fromarray(im * 255)
        pil_i.convert('L').save(os.path.join(folder, fname))
    file.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distances",
                        help="distances to test",
                        default=None)
    parser.add_argument("-s", "--im_size",
                        help="image size",
                        type=int, default=100)
    parser.add_argument("-N", "--N",
                        help="number of images to generate",
                        type=int, default=100)
    parser.add_argument("-c", "--n_colors",
                        help="number of colors to use",
                        type=int, default=9)
    parser.add_argument("-a", "--alphas",
                        help="exponent to use to control clutter",
                        type=float, default=3, nargs='+')
    parser.add_argument("--max_size",
                        help="maximum rectangle size, defaults to image size",
                        type=int, default=None)
    parser.add_argument("-f", "--folder",
                        help="folder to save to, also sets the csv name",
                        type=str, default='./data_test')
    parser.add_argument("--same_color",
                        help="whether and how to enforce the same color.\n"
                        + "[0=not at all, 1=color in, 2=rejection sample]",
                        choices=[0, 1, 2],
                        type=int, default=2)
    parser.add_argument("-x", "--split_folders",
                        help="split dataset into subfolders with 10**x files per folder",
                        type=int, default=0)
    parser.set_defaults(average=False)
    args = parser.parse_args()
    main(distances=args.distances, size=args.im_size, N=args.N,
         n_colors=args.n_colors, alphas=args.alphas, max_size=args.max_size,
         folder=args.folder, same_color=args.same_color, split_folders=args.split_folders)
