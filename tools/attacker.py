import numpy as np
import os, pickle, sys, time
import timeit

## import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn.functional as Func
import torch.nn as nn
from torchvision import datasets, transforms
#from BayesCNN_pytorch import BayesCNN

def fgsm_targeted(predections, x_original, eps, clip_min = None, clip_max=None, target_class = 0):
    loss = torch.mean(torch.log(predections[:, target_class])) # not sum?
    loss.backward()
    x_adv = x_original + eps * torch.sign(x_original.grad)
    if clip_min != None:
        x_adv = torch.clamp(x_adv, min = clip_min)
    if clip_max != None:
        x_adv = torch.clamp(x_adv, max = clip_max)
    return x_adv.data


def fgsm(predictions, x_original, eps, clip_min = None, clip_max = None):
    logp, _ = torch.max(torch.log(predictions), dim = 1)
    loss = torch.sum(logp)
    loss.backward()
    x_adv = x_original - eps * torch.sign(x_original.grad)
    if clip_min != None:
        x_adv = torch.clamp(x_adv, min = clip_min)
    if clip_max != None:
        x_adv = torch.clamp(x_adv, max = clip_max)
    return x_adv.data

'''
def model_eval(predictions, labels, adv_predictions, Y_target = None):
    if Y_target == None:
        batch_entropy = torch.squeeze(torch.sum(- predictions * torch.log(torch.clamp(predictions, min = 1e-8, max = 1.0-1e-8)), dim = -1))
        _, predicted = torch.max(adv_predictions.data, 1)
        return predicted, batch_entropy
    else:
        sys.exit("Not yet implemented")
'''

"""
Generate imputed images
"""
def adversarial_images(net, eps=0.1, batch_size=100, dataset='mnist', total=10000):
    feature_set = []
    label_set = []
    if dataset == 'mnist':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size)
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/MNIST-FASHION', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size)

    accumu_num = 0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            imagesv = Variable(images.cuda(), requires_grad=True)
            labels = labels.cuda()
        else:
            imagesv = Variable(images, requires_grad=True)
        predictions = torch.exp(net.forward_no_drop(imagesv))
        adv_images = fgsm(predictions, imagesv, eps, clip_min =0, clip_max =1)
        adv_imagesv = Variable(adv_images)
        if torch.cuda.is_available():
            adv_imagesv = Variable(adv_images.cuda())
        else:
            adv_imagesv = Variable(adv_images)
        feature_set.append(adv_imagesv)
        label_set.append(labels)
        accumu_num += labels.size()[0]
        if accumu_num >= total - 1:
            break

    return feature_set, label_set

"""
Test accuracy and entropy on these imputed images
"""
class Adversarial():
    def __init__(self, feature_sets, label_sets):
        self.features = feature_sets
        self.labels = label_sets
        self.ensembles = {}
        for eps in self.labels:
            self.ensembles[eps] = {}

    def test(self, net, if_print=1, train_mode=True):
        if train_mode:
            net.train() # dropout works at this mode
        else:
            net.eval()
        res = []
        for eps in sorted(self.labels):
            correct = 0
            total = 0
            for i in range(len(self.features[eps])):
                adv_imagesv, labels = self.features[eps][i], self.labels[eps][i]
                adv_predictions = net.forward(adv_imagesv).data
                if i not in self.ensembles[eps]:
                    self.ensembles[eps][i] = 0
                self.ensembles[eps][i] += adv_predictions
                total += labels.size(0)
                predicted = self.ensembles[eps][i].max(1)[1]
                correct += (predicted == labels).sum().item()

            res.append(correct * 1.0 / total)
        print 'Adversarial Accuracy'
        print res


def adversarial_bma(net, feature_set, label_set, predictions, bayes_type, repeats=10, if_print=1):
    net.train() # dropout works at this mode
    correct, total = 0, 0
    for i in range(len(feature_set)):
        adv_imagesv, labels = feature_set[i], label_set[i]
        adv_predictions = 0
        for _ in range(repeats):
            if bayes_type == 'vanilla':
                adv_predictions += net.forward_no_drop(adv_imagesv).data
            else:
                adv_predictions += net.forward(adv_imagesv).data
        prediction = adv_predictions.max(1)[1]
        correct += prediction.eq(labels).sum()
        total += labels.size(0)
    accuracy = 100.0 * correct / total
    if if_print:
        print 'Adversarial Ensemble Accuracy:', accuracy
    return accuracy
