from __future__ import absolute_import

'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import sys
import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal, Laplace


__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) / 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_hidden(self, pars):
        self.a = pars.a
        self.b = pars.b
        self.nu = pars.nu
        self.lamda = pars.lamda
        self.v0 = pars.v0
        self.v1 = pars.v1
        self.target_sparse = pars.sparse
        self.adaptive_sparse = 0
        self.sd = pars.sd
        self.theta = {}
        self.p_star = {}
        self.d_star = {}
        self.d_star0 = {}
        self.d_star1 = {}
        self.prior = pars.method
        self.wdecay = pars.wdecay
        self.dcoef = {'c': pars.dc, 'A': pars.da, 't': 1.0, 'alpha': pars.dalpha}
        self.total_no_pars = 0
        self.cut = pars.cut
        self.N = pars.N
        self.finetune = pars.finetune

        for name, param in self.named_parameters():
            self.total_no_pars += np.prod(param.size())
            if name.endswith('weight') and 'conv' in name and name != 'conv1.weight':
                self.p_star[name] = torch.cuda.FloatTensor(param.data.size()).fill_(0.5)
                self.d_star[name] = torch.cuda.FloatTensor(param.data.size()).fill_(5e-4)
                self.d_star0[name] = torch.cuda.FloatTensor(param.data.size()).fill_(5e-4)
                self.d_star1[name] = torch.cuda.FloatTensor(param.data.size()).fill_(5e-4) 
                self.theta[name] = pars.theta



    """
    See below how to make L2 autograd
        https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951
    For details of batch normalization
        https://pytorch.org/docs/0.3.1/nn.html    # torch.nn.BatchNorm2d
        https://discuss.pytorch.org/t/how-do-i-set-weights-of-the-batch-normalization-layer-of-pytorch/3490 # meanings of bn parameters
        https://discuss.pytorch.org/t/4d-1d-tensor-product/13438  # how to recover the normalized weights
    Mixture of Gaussian-Laplace update
    """
    def cal_nlpos(self, nlloss):
        # cross-entropy is averaged loss
        for name, param in self.named_parameters():
            if name.endswith('weight') and 'conv' in name and name != 'conv1.weight' and self.finetune > 0:
                # cross-entropy is averaged loss, we also modify the priors accordingly
                nlloss += (torch.sum(param.pow(2) * self.d_star1[name]) / self.sd**2 / 2 + torch.sum(param.abs() * self.d_star0[name]) / self.sd) / self.N
            else:
                nlloss += 0.5 * self.wdecay * param.norm(2)**2
        return nlloss

    def update_hidden(self, prune=False, adaptive_sparse=False):
        self.dcoef['t'] = self.dcoef['t'] + 1
        if self.adaptive_sparse < self.target_sparse:
            self.adaptive_sparse = self.target_sparse * (1 - self.cut**(self.dcoef['t']/50.))
        self.decay = self.dcoef['c'] / math.pow(self.dcoef['A'] + self.dcoef['t'], self.dcoef['alpha'])
        sparse_items = 0
        neg_B = 0
        neg_four_AC = 0
        num_pars = 0
        for name, param in self.named_parameters():
            if not name.endswith('weight') or  'conv' not in name or name == 'conv1.weight':
                sparse_items += (param.data == 0).sum().item()
                continue
            layer_no_pars = np.prod(param.data.size())
            if self.prior == 'ssgl':
                a_star = Normal(torch.tensor([0.0], device='cuda'), np.sqrt(self.v1)).log_prob(param.data).exp() * self.theta[name]
                b_star = Laplace(torch.tensor([0.0], device='cuda'), self.v0).log_prob(param.data).exp() * (1 - self.theta[name])
                self.p_star[name] = (1 - self.decay) * self.p_star[name] + self.decay * a_star / (a_star + b_star)
                self.d_star0[name] = (1 - self.decay) * self.d_star0[name] + self.decay * ((1 - self.p_star[name]) / self.v0)
                self.d_star1[name] = (1 - self.decay) * self.d_star1[name] + self.decay * (self.p_star[name] / self.v1)
                self.theta[name] = (1 - self.decay) * self.theta[name] + self.decay * ((self.p_star[name].sum() + self.a - 1) / (self.a + self.b + layer_no_pars - 2)).item()
                num_pars += layer_no_pars
                kept_ratio = (self.p_star[name] > 0.5).sum().item() * 100.0 / layer_no_pars
            if prune:
                threshold = self.binary_search_threshold(param.data, self.adaptive_sparse, layer_no_pars)
                
                mask = abs(param.data) < threshold
                param.data[mask] = 0
                if self.prior == 'ssgl':
                    param.data[mask] = 0
                    neg_B +=  (param.data.abs() * self.d_star0[name]).sum().item()
                    neg_four_AC += (param.data**2 * self.d_star1[name]).sum().item()
            
            if self.dcoef['t'] % 50 == 0:
                if self.prior == 'ssgl':
                    print('{:s} | Theta: {:5.1f}% | P star avg: {:5.1f} max: {:5.1f} min: {:5.1f} | L2 avg: {:0.1e} Max: {:0.1e} Min: {:0.1e} | L1 avg: {:0.1e}  Max: {:0.1e} Min: {:0.1e} | kept ratio: {:5.1f}% | SD: {:0.2e}'.format(name, self.theta[name] * 100, self.p_star[name].mean() * 100, self.p_star[name].max() * 100, self.p_star[name].min() * 100, self.d_star1[name].mean().item()/self.sd**2 / self.N, self.d_star1[name].max().item()/self.sd**2 / self.N, self.d_star1[name].min().item()/self.sd**2/ self.N, self.d_star0[name].mean().item()/self.sd / self.N, self.d_star0[name].max().item()/self.sd / self.N, self.d_star0[name].min().item()/self.sd / self.N, kept_ratio, self.sd))
            
            sparse_items += (param.data == 0).sum().item()
        self.sparse_rate = sparse_items * 100.0 / self.total_no_pars
        neg_four_AC = 4 * (num_pars + self.nu + 2) * (neg_four_AC + self.nu * self.lamda)
        new_sd = (neg_B + np.sqrt(neg_B**2 + neg_four_AC))/(num_pars + self.nu + 2)/2
        self.sd = np.sqrt((1 - self.decay) * self.sd ** 2 + self.decay * (new_sd**2))
    
    def binary_search_threshold(self, param, target_percent, total_no):
        l, r= 0., 1e2
        while l < r:
            mid = (l + r) / 2
            sparse_items = (abs(param) < mid).sum().item() * 1.0
            sparse_rate = sparse_items / total_no
            if abs(sparse_rate - target_percent) < 0.002:
                return mid
            elif sparse_rate > target_percent:
                r = mid
            else:
                l = mid
        return mid


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
