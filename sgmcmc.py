''' Adaptive SGHMC
An Adaptive Empirical Bayesian Method for Sparse Deep Learning (NeurIPS 2019)
(c) Wei Deng, Xiao Zhang, Faming Liang, Guang Lin
'''

import sys
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, pars, criterion):
        self.net = net
        self.eta = pars.lr
        self.momentum = pars.momentum
        self.invT = pars.invT
        self.wdecay = pars.wdecay
        self.V = 1.
        self.velocity = []
        self.criterion = criterion

        self.beta = 0.5 * self.V * self.eta
        self.alpha = 1 - self.momentum
        
        if self.beta > self.alpha:
            sys.exit('Momentum is too large')
        
        self.sigma = np.sqrt(2.0 * self.eta * (self.alpha - self.beta))
        self.scale = self.sigma / np.sqrt(self.invT)

        for param in net.parameters():
            p = torch.zeros_like(param.data)
            self.velocity.append(p)
    

    def backprop(self, x, y, types='pos'):
        self.net.zero_grad()
        loss = self.criterion(self.net(x), y)
        if types == 'pos':
            loss = self.net.cal_nlpos(loss)
        loss.backward(retain_graph=True)
        """ transform cross entropy to negative log likelihood """
        return loss

    def step(self, x, y):
        loss = self.backprop(x, y, 'pos')
        for i, (name, param) in enumerate(self.net.named_parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(self.scale)
            grads = param.grad.data
            self.velocity[i].mul_(self.momentum).add_(-self.eta, grads).add_(proposal)
            param.data.add_(self.velocity[i])
        return loss

    def sgd(self, x, y):
        loss = self.backprop(x, y, 'nll')
        for i, param in enumerate(self.net.parameters()):
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(self.wdecay, param.data)
            if self.momentum != 0:
                self.velocity[i].mul_(self.momentum).add_(grads)
                grads = self.velocity[i]
            param.data.add_(-self.eta, grads)
        return loss

