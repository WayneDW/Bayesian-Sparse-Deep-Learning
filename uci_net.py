import math, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Laplace
from torch.autograd import Variable as V


class Net(torch.nn.Module):
    def __init__(self, features, hidden_node):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(features, hidden_node)
        self.predict = torch.nn.Linear(hidden_node, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.predict(x)
        return x

class BayesNet(Net):
    def __init__(self, features, hidden_node, total_samples, criterion, pars):
        Net.__init__(self, features, hidden_node)
        self.criterion = criterion
        self.c = pars.c
        self.N, self.batch = total_samples, pars.batch
        """ tuning parameters """
        self.v0, self.v1 = pars.v0, pars.v1
        """ informative priors: Beta, Gamma """
        self.sd, self.wdecay = pars.sd, pars.wdecay
        self.a, self.b = pars.a, pars.b
        self.nu, self.lamda = pars.nu, pars.lamda
        self.theta, self.p_star, self.d_star, self.d_star0, self.d_star1, self.mask = {}, {}, {}, {}, {}, {}
        self.thres, self.warm = pars.thres, pars.warm
        self.num_pars, self._iter = 0, 0
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                self.p_star[name] = torch.FloatTensor(param.data.size()).fill_(0.5)
                self.d_star[name] = torch.FloatTensor(param.data.size()).fill_(pars.wdecay)
                self.d_star0[name] = torch.FloatTensor(param.data.size()).fill_(pars.wdecay)
                self.d_star1[name] = torch.FloatTensor(param.data.size()).fill_(pars.wdecay)
                self.theta[name] = 0.5
                self.mask[name] = param.data < -1e8
                self.num_pars += np.prod(param.data.size())

    def cal_nlpos(self, outputs, labels):
        nlloss = self.criterion(outputs, torch.unsqueeze(labels, dim=1)) * self.N / self.batch
        self.likelihood = nlloss.item()
        for name, param in self.named_parameters():
            if self.c == 'sghmc':
                nlloss += 0.5 * self.wdecay * param.norm(2)**2
            else:
                if name.endswith('weight'):
                    nlloss += torch.sum(param.pow(2) * self.d_star1[name].data) / self.sd**2 / 2 \
                            + torch.sum(param.abs() * self.d_star0[name].data) / self.sd
                else:
                    nlloss += 0.5 * self.wdecay * param.norm(2)**2
        return nlloss
    
    def update_decay(self):
        if self.c == 'em':
            self.decay = 1
        elif self.c == 'sa':
            self.decay = 1. / math.pow(100 + self._iter, 0.7)
        else:
            sys.exit('Error')

    def update_hidden(self):
        self._iter = self._iter + 1
        self.update_decay()

        wlasso, wridge, num_pars = 0, 0, 0
        for name, param in self.named_parameters():
            if not name.endswith('weight'):
                continue
            a_star = Normal(torch.tensor([0.0]), np.sqrt(self.v1)).log_prob(param).exp() * self.theta[name]
            b_star = Laplace(torch.tensor([0.0]), self.v0).log_prob(param).exp() * (1 - self.theta[name])
            self.p_star[name] = (1 - self.decay) * self.p_star[name] + self.decay * a_star / (a_star + b_star)
            self.d_star0[name] = (1 - self.decay) * self.d_star0[name] + self.decay * ((1 - self.p_star[name]) / self.v0)
            self.d_star1[name] = (1 - self.decay) * self.d_star1[name] + self.decay * (self.p_star[name] / self.v1)
            self.theta[name] = (1 - self.decay) * self.theta[name] \
                    + self.decay * ((self.p_star[name].sum() + self.a - 1) / (self.a + self.b + np.prod(param.data.size()) - 2)).item()
            wlasso +=  (param.abs() * self.d_star0[name]).sum().item()
            wridge += (param.pow(2) * self.d_star1[name]).sum().item()
            if self.thres > 0 and self._iter >= self.warm:
                """ one-shot mask """
                if self._iter == self.warm:
                    self.mask[name] = self.p_star[name] < self.thres
                param.data[self.mask[name]] = 0

        wridge = 4 * (self.N + self.num_pars + self.nu) * (self.likelihood + wridge + self.nu * self.lamda)
        new_sd = (wlasso + np.sqrt(wlasso**2 + wridge))/(self.N + self.num_pars + self.nu) / 2
        self.sd = np.sqrt((1 - self.decay) * self.sd ** 2 + self.decay * (new_sd**2))
