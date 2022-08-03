import argparse
import numpy as np
import json
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import os
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import scipy.io as scio
'''
keys: f'{data}-{mode}-{i}'
    data: one of ['aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'vgg_flower']
    mode: one of ['train', 'val', 'test']
    i: the class label in the corresponding dataset

examples: 'aircraft-train-0' 'aircraft-train-1' ... 'aircraft-train-69' 
          'aircraft-val-70' 'aircraft-val-71' ... 'aircraft-val-84'
          'aircraft-test-85' 'aircraft-test-86' ... 'aircraft-test-99'

class_mean: from key to feat
label2name: from key to name

sup_cls_dict: class label of each target episode task
keys: 'task_n' 
n: the number of target episode task, [0,599]

'''

def base_class_selection(args):
    with open('../create_bert_emebdding/label2name.json') as file:
        label2name = json.load(file)
    bert_class_mean = np.load('../create_bert_emebdding/bert_mean_words_embedding.npz')
    novel_cls_dict_dir = '../novel_class/'
    all_keys = list(bert_class_mean.keys())
    query_dset = args.query
    novel_cls_dict_pth = f'{novel_cls_dict_dir}/{query_dset}.mat'
    novel_cls_dict     = scio.loadmat(novel_cls_dict_pth)
    total_dst_names  = []
    for item in all_keys:
        dst_name = item.split('-')[0]
        if dst_name not in total_dst_names:
            total_dst_names.append(dst_name)
    print(total_dst_names)

    each_dst_tr_cls_number = {}
    for dst in total_dst_names:
        n = 0
        for key in all_keys:
            if dst in key and 'train' in key:
                n+=1
        each_dst_tr_cls_number[dst] = n
    print(each_dst_tr_cls_number)

    base_keys  = [key for key in all_keys if not('val' in key or 'test' in key)]
    # print(base_keys)
    # print(len(base_keys))
    base_bert_feat = np.array([bert_class_mean[key] for key in base_keys])
    base_bert_feat_norm  = preprocessing.normalize(base_bert_feat, norm='l2')
    base_bert_feat_norm  = base_bert_feat_norm.transpose(1, 0)
    # print(all_keys)
    total_dist_list = []
    for i in range(600):
        keys_name            = 'task_' + str(i)
        novel_class        = np.unique(novel_cls_dict[keys_name].squeeze())
        query_keys           = [query_dset+'-test-'+str(cls) for cls in novel_class]
        query_bert_feat      = np.array([bert_class_mean[key] for key in query_keys])
        query_bert_feat_norm = preprocessing.normalize(query_bert_feat, norm='l2')
        bert_dist            = np.dot(query_bert_feat_norm, base_bert_feat_norm)
        bert_dist_sum        = -bert_dist.sum(axis=0)
        dist_sum             = bert_dist_sum
        total_dist_list.append(dist_sum)

    total_dist   = np.vstack(total_dist_list)
    total_dist   = np.sum(total_dist,axis=0)
    n_class      = int(args.take_class) if args.take_class > 1 else int(args.take_class * len(base_keys))
    selected_idx = total_dist.argsort()[:n_class]

    # print(selected_idx)
    class_name_dst_tag  = []
    selected_class_name = []
    selected_labels     = []

    for idx in selected_idx:
        class_name_dst_tag.append(base_keys[idx])
        selected_class_name.append(label2name[base_keys[idx]])
        if 'aircraft' in base_keys[idx]:
            selected_labels.append(int(base_keys[idx].split('-')[-1]))
    statis_dst_select_cls_name,select_statis_number_dict = get_select_statis(selected_class_name,class_name_dst_tag)
    # create_new_dataspec_json(args,statis_dst_select_cls_name,selected_labels,each_dst_tr_cls_number)

    # total_select_number_dict = {}
    # select_statis_number_dict_keys = list(select_statis_number_dict.keys())
    # select_statis_number_dict_val  = list(select_statis_number_dict.values())

    # base_dst_names = ['ilsvrc_2012','omniglot','aircraft','cu_birds','dtd','quickdraw','fungi','vgg_flower']
    #
    # ratio_list = []
    #
    # for dset in base_dst_names:
    #     if dset in select_statis_number_dict_keys:
    #         idx = select_statis_number_dict_keys.index(dset)
    #         total_select_number_dict[dset] = select_statis_number_dict_val[idx]
    #         ratio_list.append(select_statis_number_dict_val[idx]/each_dst_tr_cls_number[dset])
    #     else:
    #         total_select_number_dict[dset] = 0
    #         ratio_list.append(0)
    #
    # print(total_select_number_dict)
    #
    # return ratio_list


def get_select_statis(selected_class_name,class_name_dst_tag):

    support_dst_names = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot', 'vgg_flower','ilsvrc_2012','quickdraw']
    statis_number_dict = {}
    statis_dst_select_cls_name = {}

    for class_name in support_dst_names:
        statis_number_dict[class_name] = 0
        statis_dst_select_cls_name[class_name] = []
        for i, selected_dst_tag in enumerate(class_name_dst_tag):
            if class_name in selected_dst_tag:
                statis_number_dict[class_name]+=1
                statis_dst_select_cls_name[class_name].append(selected_class_name[i])

    # print(statis_number_dict)
    return statis_dst_select_cls_name,statis_number_dict


def create_new_dataspec_json(args,statis_dst_select_cls_name,selected_labels,each_dst_tr_n):

    ori_dst_specs_dir    = '../data/dataset_specs/'
    base_dir             = '../selected_dataset_specs/'
    os.makedirs(base_dir,exist_ok=True)


    new_dst_specs_save_dir          = f'{base_dir}/number_{str(args.take_class)}/total/{args.query}'
    new_selected_dst_specs_save_dir = f'{base_dir}/number_{str(args.take_class)}/selected/{args.query}'

    os.makedirs(new_dst_specs_save_dir,exist_ok=True)
    os.makedirs(new_selected_dst_specs_save_dir,exist_ok=True)

    for dst_name in statis_dst_select_cls_name.keys():

        dst_spec_pth = f'{ori_dst_specs_dir}/{dst_name}_dataset_spec.json'
        with open(dst_spec_pth, 'r') as load_f:
            load_dict = json.load(load_f)
        if dst_name == 'aircraft':
            change_selected_class = list(load_dict['class_names'].values())
            new_select_class   = []
            for label in selected_labels:
                new_select_class.append(change_selected_class[label])

            load_dict['select_class'] = new_select_class
        else:
            load_dict['select_class'] = statis_dst_select_cls_name[dst_name]

        dst_specs_save_pth = f'{new_dst_specs_save_dir}/{dst_name}_dataset_spec.json'

        load_dict = json.dumps(load_dict, indent=2)
        with open(dst_specs_save_pth, 'w') as file:
            file.write(load_dict)
        file.close()
        threshold = int(each_dst_tr_n[dst_name]/10)
        if len(statis_dst_select_cls_name[dst_name]) > threshold and len(statis_dst_select_cls_name[dst_name]) >5:


            selected_dst_specs_save_pth = f'{new_selected_dst_specs_save_dir}/{dst_name}_dataset_spec.json'

            with open(selected_dst_specs_save_pth, 'w') as file:
                file.write(load_dict)
            file.close()
    out_domain_dst = ['traffic_sign','mscoco','mnist','cifar10','cifar100']
    for dst_name in out_domain_dst:

        dst_spec_pth = f'{ori_dst_specs_dir}/{dst_name}_dataset_spec.json'
        with open(dst_spec_pth, 'r') as load_f:
            load_dict = json.load(load_f)

        dst_specs_save_pth = f'{new_dst_specs_save_dir}/{dst_name}_dataset_spec.json'

        load_dict = json.dumps(load_dict, indent=2)
        with open(dst_specs_save_pth, 'w') as file:
            file.write(load_dict)
        file.close()

if __name__ == '__main__':
    # trg_dst_names = ['ilsvrc_2012','omniglot','aircraft','cu_birds','dtd','quickdraw','fungi','vgg_flower','traffic_sign','mscoco','mnist','cifar10','cifar100']
    trg_dst_names = ['ilsvrc_2012']
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat',   type=str, default='imagenet-net')
    parser.add_argument('--query',  type=str, default='aircraft')
    parser.add_argument('--metric', type=str, default='cosine', choices=['euclidean', 'cosine'])
    parser.add_argument('--take_class', type=float, default=500, help="used as ratio if < 1 and number if > 1")

    args = parser.parse_args()

    for dset in trg_dst_names:
        print('query dataset is {}'.format(dset))
        args = parser.parse_args()
        args.query = dset
        base_class_selection(args)


