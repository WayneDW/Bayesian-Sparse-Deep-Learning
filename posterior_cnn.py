'''Bayesian Resnet
An Adaptive Empirical Bayesian Method for Sparse Deep Learning (NeurIPS 2019)
(c) Wei Deng, Xiao Zhang, Faming Liang, Guang Lin
'''

import sys
import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal, Laplace


from models.cifar.resnet import ResNet

""" Inherit from the target Resnet model and buld the Bayesian Posterior """
class BayesPosterior(ResNet):
    def __init__(self, num_classes, depth):
        super(BayesPosterior, self).__init__(num_classes=num_classes, depth=depth)
        self.criterion = nn.CrossEntropyLoss()

    def set_hidden(self, pars):
        """ Beta prior for theta """
        self.a, self.b = pars.a, pars.b
        """ Inverse Gamma prior for sd """
        self.nu, self.lamda = pars.nu, pars.lamda
        """ Tuning parameter for Gaussian (v1) and Laplace (v0) prior """
        self.v0, self.v1 = pars.v0, pars.v1

        self.sd, self.wdecay, self.cut, self.gap, self.method = pars.sd, pars.wdecay, pars.cut, pars.gap, pars.method
        self.theta, self.p_star, self.d_star, self.d_star0, self.d_star1 = {}, {}, {}, {}, {}
        self.dcoef = {'c': pars.dc, 'A': pars.da, 't': 1.0, 'alpha': pars.dalpha}
        self.total_no_pars, self.sparse_no_pars = 0, 0
        """ total number of effective data points """
        self.N = pars.N
        self.prune = pars.prune
        for name, param in self.named_parameters():
            self.total_no_pars += np.prod(param.size())
            if name.endswith('weight') and 'conv' in name and name != 'conv1.weight':
                self.p_star[name] = torch.cuda.FloatTensor(param.data.size()).fill_(0.5)
                self.d_star[name] = torch.cuda.FloatTensor(param.data.size()).fill_(5e-4)
                self.d_star0[name] = torch.cuda.FloatTensor(param.data.size()).fill_(5e-4)
                self.d_star1[name] = torch.cuda.FloatTensor(param.data.size()).fill_(5e-4)
                self.theta[name] = pars.theta
                self.sparse_no_pars += np.prod(param.size())
        """ Sparsity should be higher in sparse layers so that the overall sparsity is matched """
        self.target_sparse, self.adaptive_sparse = floor(pars.sparse * self.total_no_pars / self.sparse_no_pars), 0

    def cal_nlpos(self, x, y):
        nlloss = self.criterion(self.forward(x), y)
        """ cross-entropy is averaged loss, we also modify the priors accordingly """
        for name, param in self.named_parameters():
            if name.endswith('weight') and 'conv' in name and name != 'conv1.weight' and self.prune > 0:
                nlloss += (torch.sum(param.pow(2) * self.d_star1[name]) / self.sd**2 / 2 \
                            + torch.sum(param.abs() * self.d_star0[name]) / self.sd) / self.N
            else:
                nlloss += 0.5 * self.wdecay * param.norm(2)**2
        return(nlloss)

    def update_decay(self):
        if self.method == 'sa':
            self.decay = self.dcoef['c'] / math.pow(self.dcoef['A'] + self.dcoef['t'], self.dcoef['alpha'])
        elif self.method == 'em':
            self.decay = 1.
        else:
            self.decay = 0.

    def update_hidden(self, prune=False, adaptive_sparse=False):
        self.dcoef['t'] = self.dcoef['t'] + 1.
        self.adaptive_sparse = self.target_sparse * (1 - self.cut ** (self.dcoef['t'] / self.gap))
        self.update_decay()

        sparse_items, wlasso, wridge = 0, 0, 0

        for name, param in self.named_parameters():
            if not name.endswith('weight') or  'conv' not in name or name == 'conv1.weight':
                sparse_items += (param.data == 0).sum().item()
                continue
            a_star = Normal(torch.tensor([0.0], device='cuda'), np.sqrt(self.v1)).log_prob(param.data).exp() * self.theta[name]
            b_star = Laplace(torch.tensor([0.0], device='cuda'), self.v0).log_prob(param.data).exp() * (1 - self.theta[name])
            self.p_star[name] = (1 - self.decay) * self.p_star[name] + self.decay * a_star / (a_star + b_star)
            self.d_star0[name] = (1 - self.decay) * self.d_star0[name] + self.decay * ((1 - self.p_star[name]) / self.v0)
            self.d_star1[name] = (1 - self.decay) * self.d_star1[name] + self.decay * (self.p_star[name] / self.v1)
            self.theta[name] = (1 - self.decay) * self.theta[name] + self.decay * \
                    ((self.p_star[name].sum() + self.a - 1) / (self.a + self.b + np.prod(param.data.size()) - 2)).item()
            kept_ratio = (self.p_star[name] > 0.5).sum().item() * 100.0 / np.prod(param.data.size())
            if prune:
                threshold = self.binary_search_threshold(param.data, self.adaptive_sparse, np.prod(param.data.size()))
                param.data[abs(param.data) < threshold] = 0
                wlasso +=  (param.data.abs() * self.d_star0[name]).sum().item()
                wridge += (param.data**2 * self.d_star1[name]).sum().item()

            if self.dcoef['t'] % 500 == 0:
                print('{:s} | P max: {:5.1f} min: {:5.1f}'.format(name, self.p_star[name].max() * 100, self.p_star[name].min() * 100))
            sparse_items += (param.data == 0).sum().item()

        self.sparse_rate = sparse_items * 100.0 / self.total_no_pars
        wridge = 4 * (self.sparse_no_pars + self.nu + 2) * (wridge + self.nu * self.lamda)
        new_sd = (wlasso + np.sqrt(wlasso**2 + wridge))/(self.sparse_no_pars + self.nu + 2) / 2
        self.sd = np.sqrt((1 - self.decay) * self.sd ** 2 + self.decay * (new_sd**2))

    def binary_search_threshold(self, param, target_percent, total_no):
        l, r= 0., 1e2
        while l < r:
            mid = (l + r) / 2
            sparse_items = (abs(param) < mid).sum().item() * 1.0
            sparse_rate = sparse_items / total_no
            if abs(sparse_rate - target_percent) < 0.0001:
                return mid
            elif sparse_rate > target_percent:
                r = mid
            else:
                l = mid
        return(mid)

def floor(n): return np.floor(n * 1000) / 1000
