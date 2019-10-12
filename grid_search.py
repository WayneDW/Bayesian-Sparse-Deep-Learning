#!/usr/bin/python 
import random 
import os
import time
import sys
 
secure_random = random.SystemRandom()

if len(sys.argv) == 2:
    gpu = sys.argv[1]
elif len(sys.argv) > 2:
    sys.exit('Unknown input')
else:
    gpu = '0'

for _ in range(3):
    #seed = str(random.randint(1, 10**6))
    seed = secure_random.choice(['125894', '130331', '258855', '421348', '248702', '248702', '248702'])
    v0 = secure_random.choice(['0.005'])
    v1 = secure_random.choice(['1e-5'])
    sd = secure_random.choice(['1'])
    lr = secure_random.choice(['1e-4'])
    invT = secure_random.choice(['1e8', '5e8', '1e9'])
    anneal = secure_random.choice(['1.005', '1.007', '1.01'])
    NN = secure_random.choice(['25000'])
    sn = '1000'
    sparse = '0.899'
    cut = secure_random.choice(['0.99'])
    #os.system('python bayes_cnn.py -prune 0 -save 1 -lr 0.1 -N ' + NN + ' -sn ' + sn + ' -v0 ' + v0 + ' -v1 ' + v1 + ' -seed ' + seed + ' -sparse 0 ' + ' -sd ' + sd + ' -invT 1e9 -gpu ' + gpu + ' -anneal 1 ' + ' > ./output/resnet20_cifar10_sn_' + sn + '_N_' + NN + '_v0_' + v0 + '_v1_' + v1 + '_sd_' + sd + '_invT_' + invT + '_anneal_1_pretrain_rand_' + seed)
    os.system('python bayes_cnn.py -lr ' + lr + ' -prune 1 -N ' + NN + ' -sn ' + sn + ' -v0 ' + v0 + ' -v1 ' + v1 + ' -seed ' + seed + ' -sparse ' + sparse + ' -sd ' + sd + ' -invT ' + invT + ' -gpu ' + gpu + ' -anneal ' + anneal + ' > ./output/resnet20_cifar10_lr_' + lr + '_sn_' + sn + '_N_' + NN + '_v0_' + v0 + '_v1_' + v1 + '_sd_' + sd + '_invT_' + invT + '_anneal_' + anneal + '_sparse_' + sparse + '_rand_' + seed)
