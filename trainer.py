''' Trainer file
An Adaptive Empirical Bayesian Method for Sparse Deep Learning (NeurIPS 2019)
(c) Wei Deng, Xiao Zhang, Faming Liang, Guang Lin
'''
import math
import copy
import sys
import os
import timeit
import csv
import dill
import argparse
import random
from random import shuffle

from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np

## import pytorch modules
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

## Import helper functions
from tools import model_eval, BayesEval
from sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()


def sgmcmc(net, train_loader, test_loader, pars):
    '''
    Perform SG-MCMC, which we sample the weights and optimize the hidden variables at the same time
    '''
    net.invT = pars.invT # reset the tempreture, useful when start from pretrained model
    
    start = timeit.default_timer()
    if pars.model == 'resnet20':
        criterion = nn.CrossEntropyLoss()
        sampler = Sampler(net, pars, criterion)
        net.set_hidden(pars)
        net.sparse_rate = 0
    best_acc = 0
    counter = 1.
    for epoch in range(1, pars.sn + 1):
        # switch to train mode
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            loss = sampler.step(images, labels)
            if pars.finetune > 0:
                net.update_hidden(prune=True, adaptive_sparse=True)
        """ Anneal learning rate """
        if pars.model == 'resnet20' and epoch in [700, 900]:
            sampler.eta *= 0.1
        """ Anneal temperature """
        if pars.model == 'resnet20':
            sampler.invT *= pars.anneal
    
        acc = model_eval(net, test_loader, if_print=0)
        if net.adaptive_sparse >= net.target_sparse - 0.005:
            best_acc = max(best_acc, acc)
        print('\nEpoch {} Sparse Rate: {:.2f}% Acc: {:0.2f} Best Acc: {:0.2f} InvT: {:.1E} Loss: {:0.1f}'.format(\
                    epoch, net.sparse_rate, acc, best_acc, sampler.invT, loss))
        if acc < 15 and epoch > 10:
            exit('Sampling lr may be too large')

    end = timeit.default_timer()
    print("Sampling Time used: {:0.1f}".format(end - start))
    if pars.sn > 0:
        model_eval(net, test_loader)

