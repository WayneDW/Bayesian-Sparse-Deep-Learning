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

for _ in range(1):
    #seed = str(random.randint(1, 10**6))
    seed = secure_random.choice(['125894'])
    dc = secure_random.choice(['0.02'])
    invT = secure_random.choice(['1e3'])
    anneal, c = secure_random.choice([('1.005', 'sa'), ('1.005', 'em'), ('1.005', 'sghmc'), ('1.0', 'sa')])
    anneal, c = ('1.005', 'sa')
    v0, v1, sparse = secure_random.choice([('0.5', '1e-3', '0.3'), ('0.1', '5e-4', '0.5'), ('0.1', '5e-5', '0.7'), ('0.005', '1e-5', '0.9')])
    v0, v1, sparse = ('0.005', '1e-5', '0.9')
    #os.system('python bayes_cnn.py -prune 0 -save 1 -lr 2e-6 ' + ' -seed ' + seed + ' -sparse 0 ' + ' -invT 1e9 -gpu ' + gpu + ' -anneal 1 ' + ' > ./output/resnet20_cifar10_invT_' + invT + '_anneal_1_pretrain_rand_' + seed)

    os.system('python bayes_cnn.py -c ' + c + ' -dc ' + dc + ' -prune 1 ' + ' -v0 ' + v0 + ' -v1 ' + v1 + ' -seed ' + seed + ' -sparse ' + sparse + ' -invT ' + invT + ' -gpu ' + gpu + ' -anneal ' + anneal + ' > ./output/resnet20_cifar10_dc_' + dc + '_v0_' + v0 + '_v1_' + v1 + '_invT_' + invT + '_anneal_' + anneal + '_sparse_' + sparse + '_' + c + '_scott_rand_' + seed)
