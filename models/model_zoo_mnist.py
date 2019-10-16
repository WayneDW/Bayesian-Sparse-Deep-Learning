#!/usr/bin/python
# from __future__ import print_function
import math
import copy
import sys
import os
import timeit
import csv
from tqdm import tqdm ## better progressbar
from math import exp

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn

from numpy import genfromtxt
from copy import deepcopy

"""
Small MNIST/F-MNIST models
"""
class CNN(nn.Module):
    def __init__(self): 
        super(CNN, self).__init__()       
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) 
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2) 
        self.fc1 = nn.Linear(64*7*7, 200) 
        self.fc2 = nn.Linear(200, 10)     

    def convs(self, x):
        x = Func.max_pool2d(Func.relu(self.conv1.forward(x)), 2)
        x = Func.max_pool2d(Func.relu(self.conv2.forward(x)), 2) 
        return(x)

    def clf(self, x): 
        x = x.view(-1, 64*7*7)    
        x = Func.relu(self.fc1.forward(x))  
        x = self.fc2.forward(x) 
        return(Func.log_softmax(x, dim=1))
    
    def forward(self, x, dropout=False):
        x = self.convs(x)
        if dropout:
            x = Func.dropout(x, p=0.5, training=self.training)
        return(self.clf(x))


"""
Mid MNIST/F-MNIST models
github.com/cmasch/zalando-fashion-mnist/blob/master/Simple_Convolutional_Neural_Network_Fashion-MNIST.ipynb
"""
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.bn_conv1 = nn.BatchNorm2d(1) 
        self.conv1 = nn.Conv2d(1, 64, 4, padding=2)    
        self.conv2 = nn.Conv2d(64, 64, 4, padding=0)  
        self.fc1 = nn.Linear(64*5*5, 256)   
        self.fc2 = nn.Linear(256, 64) 
        self.bn_fc3 = nn.BatchNorm1d(64)         
        self.fc3 = nn.Linear(64, 10) 

    def convs(self, x):
        x = Func.relu(self.conv1.forward(self.bn_conv1(x))) # BatchNorm
        x = Func.max_pool2d(x, 2) 
        x = Func.dropout(x, p=0.1, training=self.training)
        x = Func.relu(self.conv2.forward(x))  
        x = Func.max_pool2d(x, 2)    
        x = Func.dropout(x, p=0.3, training=self.training)  
        x = x.view(-1, 64*5*5) 
        x = Func.relu(self.fc1.forward(x))            
        return(x)

    def clf(self, x):
        x = Func.relu(self.bn_fc3(self.fc2.forward(x))) # BatchNorm  
        x = self.fc3.forward(x)  
        return(Func.log_softmax(x, dim=1))

    def forward(self, x, dropout=False):
        x = self.convs(x)
        if dropout:
            x = Func.dropout(x, p=0.5, training=self.training)
        return(self.clf(x))

