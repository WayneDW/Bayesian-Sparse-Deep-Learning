import math
import numpy as np
import copy
import sys
import os
import timeit
import csv
import dill
from tqdm import tqdm ## better progressbar
from math import exp
import random

import numpy as np
from numpy import genfromtxt


## import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn
from torchvision import datasets #, transforms

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
         
import torch.utils.data as data
import torchvision.datasets as datasets
 
import transforms
from copy import deepcopy
from sys import getsizeof


def loader(train_size, test_size, args):
    if args.data.startswith('cifar'):
        if args.data == 'cifar10':
            dataloader = datasets.CIFAR10
        else:
            dataloader = datasets.CIFAR100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.data == 'mnist':
        dataloader = datasets.MNIST
        transform_train = transforms.Compose([
            # https://github.com/hwalsuklee/tensorflow-mnist-cnn/blob/master/mnist_data.py
            #transforms.RandomAffine(translate=0.12),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif args.data == 'fmnist':
        dataloader = datasets.FashionMNIST
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3, mean=[0.4914]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        exit('Unknown dataset')

    if args.aug == 0:
        transform_train = transforms.ToTensor()
        transform_test = transforms.ToTensor()
        
    trainset = dataloader('./data/' + args.data.upper(), train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True, num_workers=0) # num_workers=0 is crucial for seed

    testset = dataloader(root='./data/' + args.data.upper(), train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, dataloader


class BayesEval:
    def __init__(self):
        self.counter = 0
        self.bma = []
        self.cur_acc = 0
        self.bma_acc = 0
        self.best_cur_acc = 0
        self.best_bma_acc = 0

    def eval(self, net, data_loader, weight=1, bma=False, burnIn=100):
        net.eval()
        one_correct, bma_correct = 0, 0
        self.counter += 1

        for cnt, (images, labels) in enumerate(data_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net.forward(images).data
            one_correct += outputs.max(1)[1].eq(labels.data).sum().item()
            if bma == True and self.counter >= burnIn:
                outputs = torch.exp(outputs) * weight
                if self.counter == burnIn:
                    self.bma.append(outputs)
                else:
                    self.bma[cnt] += outputs
                prediction = self.bma[cnt].max(1)[1]
                bma_correct += prediction.eq(labels.data).sum().item()
        
        self.cur_acc = 100.0 * one_correct / len(data_loader.dataset)
        self.bma_acc = 100.0 * bma_correct / len(data_loader.dataset)
        self.best_cur_acc = max(self.best_cur_acc, self.cur_acc)
        self.best_bma_acc = max(self.best_bma_acc, self.bma_acc)

def model_eval(net, data_loader, epoch=0, if_print=1):
    net.eval() 
    correct = 0 
    total = 0 
    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net.forward(images)
        prediction = outputs.data.max(1)[1]
        correct += prediction.eq(labels.data).sum().item()
    if if_print:
        print 'Epoch {} Test set accuracy: {:0.2f}%'.format(\
            epoch, 100.0 * correct / len(data_loader.dataset))
    return(100.0 * correct / len(data_loader.dataset))

def save_or_pretrain(net, num_epochs, model_name):
    if num_epochs > 0:
        print('Save model')
        torch.save(net.state_dict(), model_name)
    else:
        print('Use preTrained model')
        try:
            net.load_state_dict(torch.load(model_name))
        except:
            net.load_state_dict(torch.load(model_name, map_location='cuda:0'))
    return net
    
