import subprocess
import os


trgset            = ['ilsvrc_2012','cu_birds','dtd','quickdraw','fungi','vgg_flower','omniglot','aircraft','mscoco','traffic_sign','mnist','cifar10','cifar100']

dataspec_root_dir    = './dataset_specs/'
dump_size            = 600
mode                 = 'test'
novel_class_save_dir = '../novel_class/'
for dataset in trgset:

    cmd = 'python get_novel_class.py'
    cmd += ' --data.train ' + dataset

    cmd += ' --data.val '               + dataset
    cmd += ' --data.test '              + dataset
    cmd += ' --data.trgset '            + dataset
    cmd += ' --data.dataspec_root_dir ' + dataspec_root_dir
    cmd += ' --data.novel_class_save_dir '  + novel_class_save_dir
    cmd += ' --dump.size='              + '%d'% int(dump_size)
    cmd += ' --dump.mode='              + mode


    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
