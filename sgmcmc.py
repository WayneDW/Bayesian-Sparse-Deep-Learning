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
    def __init__(self, net, pars):
        self.net = net
        self.eta = pars.lr
        self.momentum = pars.momentum
        self.invT = pars.invT
        self.wdecay = pars.wdecay
        self.V = 1.
        self.velocity = []
        self.cuda = pars.cuda

        self.beta = 0.5 * self.V * self.eta
        self.alpha = 1 - self.momentum
        
        if self.beta > self.alpha:
            sys.exit('Momentum is too large')
        
        for param in net.parameters():
            p = torch.zeros_like(param.data)
            self.velocity.append(p)

    def backprop(self, x, y):
        self.net.zero_grad()
        loss = self.net.cal_nlpos(x, y)
        loss.backward(retain_graph=True)
        return loss

    def step(self, x, y):
        loss = self.backprop(x, y)
        """ adjust learning rate and temperature """
        sigma = np.sqrt(2.0 * self.eta * (self.alpha - self.beta))
        scale = sigma / np.sqrt(self.invT)
        for i, (name, param) in enumerate(self.net.named_parameters()):
            if self.cuda:
                proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(scale)
            else:
                proposal = torch.FloatTensor(param.data.size()).normal_().mul(scale)
            grads = param.grad.data
            self.velocity[i].mul_(self.momentum).add_(-self.eta, grads).add_(proposal)
            param.data.add_(self.velocity[i])
        return loss

    def sgd(self, x, y):
        loss = self.backprop(x, y)
        for i, param in enumerate(self.net.parameters()):
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(self.wdecay, param.data)
            if self.momentum != 0:
                self.velocity[i].mul_(self.momentum).add_(grads)
                grads = self.velocity[i]
            param.data.add_(-self.eta, grads)
        return loss

