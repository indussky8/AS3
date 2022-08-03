import subprocess
import os

query_number      = 500
strategy          = 'bert_based'
trgset            = ['dtd']
model_save_dir    = f'/mfs/xxzhang/Fewshot_Selection/save_model/{strategy}/number_based_{str(query_number)}/'
selected_dataspec_root_dir = f'/mfs/xxzhang/Fewshot_Selection/select_splits/{strategy}/number_based_{str(query_number)}/selected/'
dataspec_root_dir = f'/mfs/xxzhang/Fewshot_Selection/select_splits/{strategy}/number_based_{str(query_number)}/original/'
backbone          = 'resnet18_sel'
mode              = 'test'

for dataset in trgset:  # Plz use a larger sample.

    cmd  = 'CUDA_VISIBLE_DEVICES=0 python test_ds3.py'
    cmd += ' --data.train '

    dst_spec_list = os.listdir(f'{selected_dataspec_root_dir}/{dataset}')
    support_dst_names = []
    for i, item in enumerate(dst_spec_list):
        dset = item.split('_')[0]

        if dset == 'vgg' or dset == 'cu':
            support_dst_names.append(dset+'_'+item.split('_')[1])
        elif dset == 'ilsvrc':
            support_dst_names.append(dset+'_'+'2012')
        else:
            support_dst_names.append(dset)

    for i, dset in enumerate(support_dst_names):
        cmd += dset
        if i != (len(dst_spec_list)-1):
            cmd += ' '

    cmd += ' --data.val '               + dataset
    cmd += ' --data.test '              + dataset
    cmd += ' --data.trgset '            + dataset
    cmd += ' --data.dataspec_root_dir ' + dataspec_root_dir
    cmd += ' --model.backbone='         + backbone
    cmd += ' --model.save_dir=' + model_save_dir
    cmd += ' --dump.mode=' + mode

    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
