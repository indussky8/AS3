import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from scipy.special import softmax
from sklearn import preprocessing
from data.lmdb_dataset import LMDBDataset
from models.models_dict import DATASET_MODELS_DICT
from config import args
from ds3_utils import ds3
import scipy.io as scio
def compute_prototype(support_feas,support_labels):
    unique_labels = np.unique(support_labels)
    n_category    = unique_labels.shape[0]
    prots         = np.zeros((n_category,support_feas.shape[1]))

    for i in range(n_category):
        idx       = np.where(support_labels == i)[0]

        prots[i,:] = support_feas[idx, :].mean(0)

    return prots

all_support_dataset = ['ilsvrc_2012','cu_birds','dtd','quickdraw','fungi','vgg_flower','omniglot','aircraft']

def main():
    LIMITER = 600

    # Setting up datasets
    dataspec_root_dir = args['data.dataspec_root_dir']
    all_test_datasets = args['data.trgset']
    extractor_domains = args['data.train']


    dump_name = args['dump.name'] if args['dump.name'] else 'test_dump'
    testset   = LMDBDataset(args,extractor_domains, all_test_datasets,
                          args['model.backbone'], 'test', dump_name, LIMITER)

    # define the embedding method
    dataset_models = DATASET_MODELS_DICT[args['model.backbone']]
    accs_names = ['AS3  ']
    all_accs   = dict()
    # Go over all test datasets
    for test_dataset in all_test_datasets:
        # print(test_dataset)
        testset.set_sampling_dataset(test_dataset)
        test_loader = DataLoader(testset, batch_size=None, batch_sampler=None, num_workers=16)
        all_accs[test_dataset] = {name: [] for name in accs_names}
        i = 0
        all_selected_weights = []
        for sample in tqdm(test_loader):
            context_labels = sample['context_labels'].numpy()
            target_labels  = sample['target_labels'].numpy()

            context_features_dict       = {k: v.numpy() for k, v in sample['context_feature_dict'].items()}
            target_features_dict        = {k: v.numpy() for k, v in sample['target_feature_dict'].items()}

            learner_weight,all_prots   = ds3(context_features_dict,context_labels)
            target_features_dict_keys  = list(target_features_dict.keys())
            all_selected_trg_feas      = []
            all_selected_prots         = []
            weights = np.zeros(8)
            for i in range(len(target_features_dict_keys)):

                # print(learner_weight.shape)
                selected_query_feas   = target_features_dict[target_features_dict_keys[i]]
                selected_prototypes   = all_prots[i]

                selected_query_feas   = preprocessing.normalize(selected_query_feas,norm='l2')
                selected_prototypes   = preprocessing.normalize(selected_prototypes,norm='l2')

                all_selected_trg_feas.append(learner_weight[i] * selected_query_feas)
                all_selected_prots.append(learner_weight[i] * selected_prototypes)

                idx = all_support_dataset.index(target_features_dict_keys[i])
                weights[idx] = learner_weight[i]



            selected_query_feas    = np.hstack(all_selected_trg_feas)
            selected_support_prots = np.hstack(all_selected_prots)
            # print(selected_idxs)

            selected_support_prots = selected_support_prots.transpose([1,0])
            logits = np.dot(selected_query_feas, selected_support_prots)
            # logits = np.reshape(logits,[-1,logits.shape[-1]])
            probs  = softmax(logits, axis=1)
            preds = probs.argmax(1)
            final_acc = np.mean(np.equal(preds, target_labels))
            all_accs[test_dataset]['AS3'].append(final_acc)
            all_selected_weights.append(weights)

    # Make a nice accuracy table

    all_selected_weights = np.vstack(all_selected_weights)
    results_save_dir = f'{args["model.save_dir"]}/'
    rows = []
    for dataset_name in all_test_datasets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(all_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            mean_acc_round = round(mean_acc, 2)
            conf_round = round(conf, 2)
            save_pth        = f'{results_save_dir}/{dataset_name}/{dataset_name}_ds3_results.txt'
            weight_save_pth = f'{results_save_dir}/{dataset_name}/{dataset_name}_weights.mat'
            scio.savemat(weight_save_pth,{'support_dset':all_support_dataset,'weight':all_selected_weights})

            with open(save_pth, 'w') as f:
                f.write(dataset_name)
                f.write('\n')
                f.write(str(mean_acc_round))
                f.write('\n')
                f.write(str(conf_round))
                f.write('\n')

            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()