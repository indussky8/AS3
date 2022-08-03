import subprocess
import os

number   = 500
trgset   = ['aircraft']

model_save_dir    = f'../save_model/number_{str(number)}/'
selected_dataspec_root_dir = f'../selected_dataset_specs/number_{str(number)}/selected/'
dataspec_root_dir = f'../selected_dataset_specs/number_{str(number)}/total/'
backbone          = 'resnet18_sel'
dump_size         = 600
mode              = 'test'
for dataset in trgset:



    cmd = 'CUDA_VISIBLE_DEVICES=2 python create_features_db.py'
    cmd += ' --data.train '

    dst_spec_list = os.listdir(f'{selected_dataspec_root_dir}/{dataset}')

    support_dst_names = []
    for i, item in enumerate(dst_spec_list):
        dset = item.split('_')[0]

        if dset == 'vgg' or dset == 'cu':
            support_dst_names.append(dset + '_' + item.split('_')[1])
        elif dset == 'ilsvrc':
            support_dst_names.append(dset + '_' + '2012')
        else:
            support_dst_names.append(dset)
    for i, item in enumerate(support_dst_names):
        dset = item.split('_')[0]
        if dset == 'vgg' or dset == 'cu':
            dset += '_'
            dset += item.split('_')[1]
        if dset == 'ilsvrc':
            dset = dset + '_' + '2012'
        cmd += dset
        if i != (len(support_dst_names)-1):
            cmd += ' '


    cmd += ' --data.val '               + dataset
    cmd += ' --data.test '              + dataset
    cmd += ' --data.trgset '            + dataset
    cmd += ' --data.dataspec_root_dir ' + dataspec_root_dir

    cmd += ' --dump.size='              + '%d'% int(dump_size)
    cmd += ' --dump.mode='              + mode

    cmd += ' --model.backbone='         + backbone
    cmd += ' --model.save_dir=' + model_save_dir

    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
