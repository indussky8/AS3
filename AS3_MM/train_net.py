#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import gin.tf

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader,
                                      MetaDatasetEpisodeReader)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from config import args



def train():
    # initialize datasets and loaders
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    dataspec_root_dir = args['data.dataspec_root_dir']
    dataset_spec_dir = os.path.join(dataspec_root_dir,args['data.trgset'][0])
    train_loader = MetaDatasetBatchReader('train',dataset_spec_dir, trainsets, valsets, testsets,
                                          batch_size=args['train.batch_size'])
    val_loader = MetaDatasetEpisodeReader('val', dataset_spec_dir,trainsets, valsets, testsets)

    # initialize model and optimizer
    num_train_classes = train_loader.num_classes('train')
    model             = get_model(num_train_classes, args)
    # print(model.device())
    optimizer         = get_optimizer(model, args, params=model.get_parameters())
    # print(num_train_classes)

    checkpointer = CheckPointer(args, model, optimizer=optimizer)
    # if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
    #     start_iter, best_val_loss, best_val_acc =\
    #         checkpointer.restore_model(ckpt='last')
    # else:
    #     print('No checkpoint restoration')
    #     best_val_loss = 999999999
    #     best_val_acc = start_iter = 0
    print('No checkpoint restoration')
    best_val_loss = 999999999
    best_val_acc = start_iter = 0

    # define learning rate policy
    if args['train.lr_policy'] == "step":
        lr_manager = UniformStepLR(optimizer, args, start_iter)
    elif "exp_decay" in args['train.lr_policy']:
        lr_manager = ExpDecayLR(optimizer, args, start_iter)
    elif "cosine" in args['train.lr_policy']:
        lr_manager = CosineAnnealRestartLR(optimizer, args, start_iter)

    # defining the summary writer
    writer = SummaryWriter(checkpointer.model_path)
    # tf.compat.v1.disable_eager_execution()



    # Training loop
    max_iter   = args['train.max_iter']
    epoch_loss = {name: [] for name in trainsets}
    epoch_acc  = {name: [] for name in trainsets}
    config     = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as session:
        for i in tqdm(range(max_iter)):
            if i < start_iter:
                continue

            optimizer.zero_grad()

            sample = train_loader.get_train_batch(session)
            logits = model.forward(sample['images'])
            batch_loss, stats_dict, _ = cross_entropy_loss(logits, sample['labels'])
            batch_dataset = sample['dataset_name']
            epoch_loss[batch_dataset].append(stats_dict['loss'])
            epoch_acc[batch_dataset].append(stats_dict['acc'])

            batch_loss.backward()
            optimizer.step()
            lr_manager.step(i)

            if (i + 1) % 200 == 0:
                for dataset_name in trainsets:
                    writer.add_scalar(f"loss/{dataset_name}-train_acc",
                                      np.mean(epoch_loss[dataset_name]), i)
                    writer.add_scalar(f"accuracy/{dataset_name}-train_acc",
                                      np.mean(epoch_acc[dataset_name]), i)
                    epoch_loss[dataset_name], epoch_acc[dataset_name] = [], []

                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], i)

            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:
                model.eval()
                dataset_accs, dataset_losses = [], []
                for valset in valsets:
                    val_losses, val_accs = [], []
                    for j in tqdm(range(args['train.eval_size'])):
                        with torch.no_grad():
                            sample = val_loader.get_validation_task(session, valset)
                            context_features = model.embed(sample['context_images'])
                            target_features = model.embed(sample['target_images'])
                            context_labels = sample['context_labels']
                            target_labels = sample['target_labels']
                            _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                              target_features, target_labels)
                        val_losses.append(stats_dict['loss'])
                        val_accs.append(stats_dict['acc'])

                    # write summaries per validation set
                    dataset_acc, dataset_loss = np.mean(val_accs) * 100, np.mean(val_losses)
                    dataset_accs.append(dataset_acc)
                    dataset_losses.append(dataset_loss)
                    writer.add_scalar(f"loss/{valset}/val_loss", dataset_loss, i)
                    writer.add_scalar(f"accuracy/{valset}/val_acc", dataset_acc, i)
                    print(f"{valset}: val_acc {dataset_acc:.2f}%, val_loss {dataset_loss:.3f}")

                # write summaries averaged over datasets
                avg_val_loss, avg_val_acc = np.mean(dataset_losses), np.mean(dataset_accs)
                writer.add_scalar(f"loss/avg_val_loss", avg_val_loss, i)
                writer.add_scalar(f"accuracy/avg_val_acc", avg_val_acc, i)

                # saving checkpoints
                if avg_val_acc > best_val_acc:
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    es = 0
                    print('Best model so far!')
                else:
                    is_best = False
                    es += 1

                    if es > 3:
                        print("Early stopping best_val_acc for this epoch: ", best_val_acc,
                              "...")
                        break

                checkpointer.save_checkpoint(i, best_val_acc, best_val_loss,
                                             is_best, optimizer=optimizer,
                                             state_dict=model.get_state_dict())

                model.train()
                print(f"Trained and evaluated at {i}")
    writer.close()
    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, best_avg_val_acc: {best_val_acc:.2f}%""")
    else:
        print(f"""No training happened. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")

    return 0

#
if __name__ == '__main__':
    print(str(train()))

