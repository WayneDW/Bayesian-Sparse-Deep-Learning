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

# (0.917, 0.005, 1e-5)

for _ in range(1):
    seed = str(random.randint(1, 10**6))
    v0 = secure_random.choice(['0.005'])
    v1 = secure_random.choice(['1e-5'])
    sd = secure_random.choice(['1'])
    invT = secure_random.choice(['1e9', '1e8', '1e7'])
    anneal = secure_random.choice(['1.005', '1.01'])
    NN = secure_random.choice(['25000'])
    sn = '1000'
    sparse = '0.9'
    prior = 'ssgl'
    cut = secure_random.choice(['0.99'])
    if prior.startswith('ssg'):
        os.system('python bayes_cnn.py -lr 0.1 -N ' + NN + ' -anneal ' + anneal + ' -sn ' + sn + ' -v0 ' + v0 + ' -v1 ' + v1 + ' -seed ' + seed + ' -save 1 -prune 0 ' + ' -sd ' + sd + ' -invT ' + invT + ' -gpu ' + gpu + ' > ./output/resnet20_cifar10_sn_' + sn + '_N_' + NN + '_v0_' + v0 + '_v1_' + v1 + '_sd_' + sd + '_invT_' + invT + '_anneal_' + anneal + '_sparse_' + sparse + '_rand_pretrain_' + seed)
        os.system('python bayes_cnn.py -prune 1 -N ' + NN + ' -sn ' + sn + ' -v0 ' + v0 + ' -v1 ' + v1 + ' -seed ' + seed + ' -sparse ' + sparse + ' -sd ' + sd + ' -invT ' + invT + ' -gpu ' + gpu + ' -anneal ' + anneal + ' > ./output/resnet20_cifar10_sn_' + sn + '_N_' + NN + '_v0_' + v0 + '_v1_' + v1 + '_sd_' + sd + '_invT_' + invT + '_anneal_' + anneal + '_sparse_' + sparse + '_rand_' + seed)
    else:
        os.system('python bayes_cnn.py -sn ' + sn + ' -seed ' + seed + ' -sparse ' + sparse + ' -gpu ' + gpu + ' -prior ' + prior + ' -penalty ' + penalty + ' > ./output/resnet20_SSGL_priors/resnet20_cifar10_sn_' + sn + '_' + prior + '_penalty_' + penalty + '_sparse_' + sparse + '_rand_' + seed)

    #os.system('sleep 600s')
