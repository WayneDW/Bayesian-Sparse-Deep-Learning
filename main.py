#!/usr/bin/evn python
import argparse

import random
import numpy as np
import pandas as pd
from math import sqrt

from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import torch
from torch.autograd import Variable

from net import BayesNet
from sgmcmc import Sampler
from uci import load_dataset

parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-c', default='sa', type=str, help='sa (sghmc-sa), em (sghmc-em), sghmc')
parser.add_argument('-data', default='boston', type=str, help='dataset')
parser.add_argument('-lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('-l2', default=1e-4, type=float, help='L2 penalty')
parser.add_argument('-invT', default=1, type=float, help='inverse temperature')
parser.add_argument('-anneal', default=1.003, type=float, help='annealing')
parser.add_argument('-v0', default=0.1, type=float, help='v0')
parser.add_argument('-v1', default=10, type=float, help='v1')
parser.add_argument('-a', default=1, type=float, help='a')
parser.add_argument('-b', default=10, type=float, help='b')
parser.add_argument('-nu', default=1, type=float, help='nu')
parser.add_argument('-lamda', default=1, type=float, help='lamda')
parser.add_argument('-sd', default=10, type=float, help='sd')
parser.add_argument('-thres', default=0, type=float, help='threshold p_star')
parser.add_argument('-warm', default=1000, type=int, help='warm up iterations')
parser.add_argument('-total_epochs', default=201, type=int, help='total number of epochs')
parser.add_argument('-batch', default=50, type=int, help='batch size')
parser.add_argument('-hidden', default=50, type=int, help='hidden nodes')
parser.add_argument('-seed', default=5, type=int, help='seeds')

pars = parser.parse_args()
torch.manual_seed(pars.seed)
torch.cuda.manual_seed(pars.seed)
np.random.seed(pars.seed)
random.seed(pars.seed)

X_train, y_train, X_test, y_test = load_dataset(pars.data, split_seed=pars.seed)

total_samples = len(X_train)
total_batches = total_samples // pars.batch 
cols = X_train.shape[1] 
print('Total data points {:d} total batches {:d}'.format(total_samples, total_batches))
net = BayesNet(cols, pars.hidden, total_samples, pars)

for name, param in net.named_parameters():
    print(name, param.shape)

criterion = torch.nn.MSELoss(reduction='sum')

sampler = Sampler(net, pars, criterion)

for epoch in range(pars.total_epochs):
    running_loss = 0.0
    X_train, y_train = shuffle(X_train, y_train)
    for i in range(total_batches):
        start = i * pars.batch
        end = start + pars.batch
        inputs = Variable(torch.FloatTensor(X_train[start:end]))
        labels = Variable(torch.FloatTensor(y_train[start:end]))
        outputs = net(inputs)
        if pars.c in ['sa', 'em', 'sghmc']:
            loss = sampler.step(outputs, labels)
        if pars.c in ['sa', 'em']:
            net.update_hidden()
        running_loss += loss.item()
    if epoch % 100 == 0:
        inputs = Variable(torch.FloatTensor(X_test))
        labels = Variable(torch.FloatTensor(y_test))
        outputs = net(inputs)
        test_loss = criterion(outputs, torch.unsqueeze(labels, dim=1))

        X_test = Variable(torch.FloatTensor(X_test))
        y_pred_test = net(X_test).data[:,0].numpy()
        rmse_test = sqrt(mean_squared_error(y_pred_test, y_test))
        sparse_rates = {}
        for name, param in net.named_parameters():
            if name.endswith('weight'):
                sparse_rates[name] = 1 - (param != 0).sum().item() * 1.0 / np.prod(param.shape)
        print('Epoch {} Train loss: {:.2f} RMSE: {:.2f}'.format(epoch, running_loss, rmse_test))

