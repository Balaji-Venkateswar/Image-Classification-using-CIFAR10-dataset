# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 06:29:34 2022

@author: BalajiVenkateswar

to-do:
    -fix ...
"""

# %% imports

import torch

# %% accuracy calculation

def accuracy(model, data):
    criterion = torch.nn.CrossEntropyLoss()
    total = 0
    correct = 0

    with torch.no_grad():
        for d in data:
            images, labels = d
            outputs = model(images)
            loss = criterion(outputs,labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accur = 100 * correct // total
    
    return loss, accur
    