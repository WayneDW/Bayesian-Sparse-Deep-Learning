#!/usr/bin/env python

import math
import sys
import argparse
from math import exp
from sys import getsizeof
import numpy as np
import random

## import pytorch modules
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

from tools import model_eval, save_or_pretrain, loader
from trainer import sgmcmc

from posterior_cnn import BayesPosterior

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('-method', default='sa', help='stochastic approximation (sa)/ EM (em)/ vanilla')
    parser.add_argument('-aug', default=1, type=float, help='Data augmentation or not')
    # sampling part: hidden variable update -- decay rate
    parser.add_argument('-dc', default=10, type=float, help='1st coef in decay C (A+t)^alpha')
    parser.add_argument('-da', default=1000, type=float, help='2nd coef in decay C (A+t)^alpha')
    parser.add_argument('-dalpha', default=0.75, type=float, help='3rd coef decay C (A+t)^alpha')
    # numper of optimization/ sampling epochs
    parser.add_argument('-data', default='cifar10', dest='data', help='MNIST/ Fashion MNIST/ CIFAR10/ CIFAR100')
    parser.add_argument('-model', default='resnet20', type=str, help='small/ mid/ large (resnet20) model')
    parser.add_argument('-train', default=1000, type=int, help='training batch size')
    parser.add_argument('-test', default=1000, type=int, help='testing batch size')
    parser.add_argument('-prune', default=0, type=int, help='prune from an exsiting model')
    parser.add_argument('-save', default=0, type=int, help='save the model or not')
    parser.add_argument('-sn', default=1000, type=int, help='sampling Epochs')
    # SGHMC hyperparameters
    parser.add_argument('-wdecay', default=5e-4, type=float, help='samling weight decay')
    parser.add_argument('-momentum', default=0.9, type=float, help='sampling momentum learning rate')
    parser.add_argument('-invT', default=1e9, type=float, help='inverse tempreture')
    parser.add_argument('-anneal', default=1.005, type=float, help='anneal tempreture')
    # setup for sparse coefficients
    parser.add_argument('-lr', default=2e-9, type=float, help='sampling learning rate (default for pruning)')
    parser.add_argument('-sparse', default=0.9,  type=float, help='target sparse Rate')
    parser.add_argument('-v0', default=0.005, type=float, help='v0')
    parser.add_argument('-v1', default=1e-5, type=float, help='v1')
    # informative priors
    parser.add_argument('-nu', default=1000,  type=float, help='inverse Gamma(nu/2, lamda*nu/2)')
    parser.add_argument('-lamda', default=1000,  type=float, help='inverse Gamma(nu/2, lamda*nu/2)')
    parser.add_argument('-a', default=2.7e5,  type=float, help='hyperparameter a for Beta (a, b)')
    parser.add_argument('-b', default=2.7e5,  type=float, help='hyperparameter a for Beta (a, b)')
    parser.add_argument('-theta', default=0.5, type=float, help='theta')
    parser.add_argument('-sd', default=1.0, type=float, help='default standard deviation')
    parser.add_argument('-N', default=50000, type=float, help='effevtive number of data points')
    # pruning rates
    parser.add_argument('-cut', default=0.99, type=float, help='sparse damping rate')
    parser.add_argument('-gap', default=50, type=float, help='gaps to update damping rate')

    # Resnet Architecture
    parser.add_argument('-depth', type=int, default=20, help='model depth.')

    # other settings
    parser.add_argument('-seed', default=995036, type=int, help='random Seed')
    parser.add_argument('-cuda', default=1, type=int, help='Use CUDA or not')
    parser.add_argument('-gpu', default=0, type=int, help='default GPU')
    parser.add_argument('-multi', default=0, type=int, help='multiple GPUs')

    pars = parser.parse_args()
    """ Step 0: Numpy printing setup, set GPU and Seeds """

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    if not torch.cuda.is_available():
        exit("CUDA does not exist!")

    torch.cuda.set_device(pars.gpu)

    torch.manual_seed(pars.seed)
    torch.cuda.manual_seed(pars.seed)
    np.random.seed(pars.seed)
    random.seed(pars.seed)
    torch.backends.cudnn.deterministic=True

    """ Step 1: Preprocessing """

    if pars.model.startswith('resnet'):
        no_c = 10
        net = BayesPosterior(num_classes=no_c, depth=pars.depth).cuda()
        # parallelized over multiple GPUs in the batch dimension
        if pars.multi:
            net = torch.nn.DataParallel(net).cuda()
    else:
        print('Unknown Model structure')

    """ Step 2: Load Data """
    train_loader, test_loader, targetloader = loader(pars.train, pars.test, pars)
    trainset = targetloader('./data/' + pars.data.upper(), train=True, transform=transforms.ToTensor())

    
    """ Step 3: Load the model """
    if pars.prune > 0:
        net = save_or_pretrain(net, 0, './output/pars.' + pars.data + '_' + pars.model + '_seed_' + str(pars.seed))
        model_eval(net, test_loader, pars, 'Pretrained')
    else:
        print('Sampling from scratch')
    
    """ Step 4: Bayesian Sampling """
    sgmcmc(net, train_loader, test_loader, pars)
    if pars.save:
        net = save_or_pretrain(net, 1, './output/pars.' + pars.data + '_' + pars.model + '_seed_' + str(pars.seed))

if __name__ == "__main__":
    main()
