import subprocess
import os


query_number      = 1500
trgset            = 'cifar100'

model_save_dir    = f'./save_model/number_{str(query_number)}/'
dataspec_root_dir = f'./selected_dataset_specs/number_{str(query_number)}/selected/'

all_dataset   = ['ilsvrc_2012','cu_birds','dtd','quickdraw','fungi','vgg_flower','omniglot','aircraft']
batchsize     = [64,16,32,64,32,8,16,8]
LR            = ['3e-2','3e-2','3e-2','1e-2','3e-2','3e-2',"3e-2","3e-2"]
max_iter      = [480000,50000,50000,480000,480000,50000,50000,50000]
anneal_freq   = [48000,3000,1500,48000,15000,1500,3000,3000]

i = 0
param_dict  = {}
for dataset in all_dataset:
    param_dict[dataset] = {}
    param_dict[dataset]['batchsize'] = batchsize[i]
    param_dict[dataset]['LR'] = LR[i]
    param_dict[dataset]['max_iter'] = max_iter[i]
    param_dict[dataset]['anneal_freq'] = anneal_freq[i]
    i+=1

dst_spec_list = os.listdir(f'{dataspec_root_dir}/{trgset}')

support_dst_names = []
for item in dst_spec_list:
    dset = item.split('_')[0]
    if dset == 'vgg' or dset == 'cu':
        dset += '_'
        dset += item.split('_')[1]
    if dset == 'ilsvrc':
        dset = dset + '_'+ '2012'

    support_dst_names.append(dset)


for dataset in support_dst_names:  # Plz use a larger sample.
    cmd = 'CUDA_VISIBLE_DEVICES=1 python train_net.py'
    cmd += ' --data.train='             + dataset
    cmd += ' --data.val='               + dataset
    cmd += ' --data.test='              + dataset
    cmd += ' --data.trgset='            + trgset
    cmd += ' --data.dataspec_root_dir=' + dataspec_root_dir

    cmd += ' --train.batch_size='         + '%d'% int(param_dict[dataset]['batchsize'])
    cmd += ' --train.learning_rate='      + param_dict[dataset]['LR']
    cmd += ' --train.max_iter='           + '%d'% int(param_dict[dataset]['max_iter'])
    cmd += ' --train.cosine_anneal_freq=' + '%d' % int(param_dict[dataset]['anneal_freq'])
    cmd += ' --train.eval_freq='          + '%d' % int(param_dict[dataset]['anneal_freq'])

    cmd += ' --model.save_dir=' + model_save_dir

    print(cmd)
    output = subprocess.check_output(cmd, shell=True)

