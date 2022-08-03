from load_json import *
import numpy as np
import json
import torch
from transformers import BertTokenizer, BertModel
import matplotlib
matplotlib.use('Agg')


def get_class_embeding(text,strategy):

    marked_text = "[CLS] " + text + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states    = outputs[2]  # tuple (number_layers,number_batch,number_tokens,num_embedding)
        token_embeddings = torch.stack(hidden_states, dim=0)  # tensor num_layers*num_batch*num_tokens*num_embedding
        token_embeddings = torch.squeeze(token_embeddings, dim=1) # tensor num_layers*num_tokens*num_embedding  num_batch=1
        if strategy == 'cls':
            txt_embedding    = token_embeddings[-1, 0, :]
        elif strategy == 'mean_words':
            txt_embedding = torch.mean(token_embeddings[-1,1:,:],dim=0)
    class_embedding_arr = txt_embedding.cpu().detach().numpy()

    return class_embedding_arr


if __name__ == '__main__':

    METADATASET_TS_NAMES = ['mnist','cifar10','cifar100','cu_birds', 'dtd', 'ilsvrc','vgg_flower','quickdraw','omniglot','fungi','aircraft','mscoco','traffic_sign']


    root_dataspec_dir = '../data/dataset_specs'

    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')
    model     = BertModel.from_pretrained('./bert-base-uncased/',
                                      output_hidden_states = True,).cuda()
    all_dset_embedding_dict = {}
    txt_embedding_strategy  = 'mean_words'

    all_class_name_ori_dict     = {}
    all_class_name_bert_dict    = {}
    all_class_textual_embd_dict = {}
    label2name_dict             = {}
    for dset in METADATASET_TS_NAMES:
        print(dset)
        dset_pth = f'{root_dataspec_dir}/{dset}_dataset_spec.json'
        class_names_ori,class_names_bert = get_class_names(dset, dset_pth)
        for key in list(class_names_bert.keys()):
            name              = class_names_bert[key]
            cls_embedding_arr = get_class_embeding(name, txt_embedding_strategy)
            all_class_textual_embd_dict[key] = list(cls_embedding_arr)

        for key in list(class_names_ori.keys()):

            label2name_dict[key] = class_names_ori[key]

    embedding_save_pth = f'./bert_{txt_embedding_strategy}_embedding.npz'
    label2name_save_pth = f'./label2name.json'
    label2name_dict = json.dumps(label2name_dict)
    with open(label2name_save_pth, 'w') as file:
        file.write(label2name_dict)
    file.close()
    np.savez(embedding_save_pth,**all_class_textual_embd_dict)
