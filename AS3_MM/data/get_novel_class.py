import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import torch
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tqdm import tqdm
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))
from data.meta_dataset_reader import MetaDatasetEpisodeReader, MetaDatasetBatchReader
from models.models_dict import DATASET_MODELS_DICT
from config import args
import scipy.io as scio


class DatasetWriter(object):
    def __init__(self, args, rewrite=True, write_frequency=10):
        self._mode = args['dump.mode']
        self._write_frequency = write_frequency
        self._db = None
        self.args = args
        self.trgset = args['data.trgset'][0]
        self.dataset_models = DATASET_MODELS_DICT[args['model.backbone']]
        self.support_class_save_dir = args['data.novel_class_save_dir']
        os.makedirs(self.support_class_save_dir,exist_ok=True)
        self.support_class_dict_save_pth = f'{self.support_class_save_dir}/{self.trgset}.mat'


        # print(self.dataset_models)

        dataspec_dir = f'{args["data.dataspec_root_dir"]}'
        trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
        # print(testsets)
        loader = MetaDatasetEpisodeReader(self._mode, dataspec_dir, trainsets, valsets, testsets)
        self._map_size = 50000 * 100 ** 2 * 512 * 8
        self.trainsets = trainsets

        if self._mode == 'train':
            evalset = "allcat"
            self.load_sample = lambda sess: loader.get_train_task(sess)
        elif self._mode == 'test':
            evalset = testsets[0]
            self.load_sample = lambda sess: loader.get_test_task(sess, evalset)
        elif self._mode == 'val':
            evalset = valsets[0]
            self.load_sample = lambda sess: loader.get_validation_task(sess, evalset)
        #
        dump_name = self._mode + '_dump' if not args['dump.name'] else args['dump.name']


    def encode_dataset(self, n_tasks=1000):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        support_class_dict = {}
        with tf.compat.v1.Session(config=config) as session:
            for idx in tqdm(range(n_tasks)):
                # compressing image
                sample = self.load_sample(session)

                keys_name = 'task_' + str(idx)
                keys_values = np.unique(sample['context_class_ids'].detach().cpu().numpy())
                support_class_dict[keys_name] = keys_values

            scio.savemat(self.support_class_dict_save_pth,support_class_dict)



if __name__ == '__main__':
    dr = DatasetWriter(args)
    dr.encode_dataset(args['dump.size'])
    print('Done')
