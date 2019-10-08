import sys
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, pars, criterion):
        self.net = net
        self.eta = pars.lr
        self.anneal = pars.anneal
        self.momentum = 0.9
        self.invT = pars.invT
        self.wdecay = pars.l2
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


    def backprop(self, outputs, labels):
        self.net.zero_grad()
        loss = self.net.cal_nlpos(outputs, labels, self.criterion)
        loss.backward()
        """ transform cross entropy to negative log likelihood """
        return loss

    def step(self, x, y):
        loss = self.backprop(x, y)
        self.invT = self.invT * self.anneal
        self.scale = self.sigma / np.sqrt(self.invT)
        for i, (name, param) in enumerate(self.net.named_parameters()):
            proposal = torch.FloatTensor(param.data.size()).normal_().mul(self.scale)
            grads = param.grad.data
            self.velocity[i].mul_(self.momentum).add_(-self.eta, grads).add_(proposal)
            param.data.add_(self.velocity[i])

        return loss
