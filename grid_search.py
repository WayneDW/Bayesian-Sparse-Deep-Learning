import os
import random

AllSeeds = [5, 55, 555, 5555, 6, 66, 666, 6666, 7, 77, 777, 7777, 8, 88, 888, 8888, 9, 99, 999, 9999]

for data in ['boston']:#, 'yacht', 'energy-efficiency', 'wine', 'concrete']:
    os.system('mkdir -p output/' + data)
    for c in ['sa']:#, 'em', 'sghmc']:
        for invT in [1]:
            for v0 in [0.1]:
                for anneal in [1.0, 1.003]:
                    for seed in AllSeeds:
                        os.system('nohup python main.py ' \
                                + ' -data ' + data \
                                + ' -c ' + c \
                                + ' -invT ' + str(invT) \
                                + ' -anneal ' + str(anneal) \
                                + ' -seed ' + str(seed) \
                                + ' -v0 ' + str(v0) \
                                + ' >> ./output/' + data + '/' + c \
                                + '_invT_' + str(invT) \
                                + '_anneal_' + str(anneal) \
                                + '_v0_' + str(v0) \
                                + '_seed_' + str(seed) + ' &')
                    """ adjust time for other datasets """
                    os.system('sleep 50s')
