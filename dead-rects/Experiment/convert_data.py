#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 converts files into proper format
"""

import numpy as np
import pandas as pd
import json 
import os


def main(folder, folder_out):
    subj_dirs = os.listdir(folder)
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    for i_sub in subj_dirs:
        if i_sub[0] != '.' and i_sub != 'Icon\r':
            files = os.listdir(os.path.join(folder, i_sub))
            for i_file in files:
                if i_file[0] != '.' and i_file != 'Icon\r'  \
                    and i_file[-3:] == 'npy':
                    dat = np.load(os.path.join(folder, i_sub, i_file))
                    print(dat.shape)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder",
                        help="folder to convert",
                        type=str,
                        default='resultsFrozen')
    parser.add_argument("-o", "--output",
                        help="folder to write into",
                        type=str,
                        default='resultsFrozen_converted')
    args = parser.parse_args()
    main(folder=args.folder)
