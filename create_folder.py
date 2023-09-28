# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 06:29:34 2022

@author: BalajiVenkateswar

to-do:
    -fix ...
"""

# %% imports

import datetime
import os

# %% trainfolder generation

def create_trainfolder(path: str):
    
    parent_dir = os.path.join(path, 'training')
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    mydir = os.path.join(parent_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    
    folders = ['data', 'config', 'img', 'model']
    
    p = []
    for items in folders:
        path = os.path.join(mydir, items)
        os.mkdir(path)
        p.append(path)
        
    return p


# %% test

if __name__ == '__main__':
    path = '.'
    folder = create_trainfolder(path)

