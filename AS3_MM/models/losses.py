import torch
import gin
import numpy as np

from torch import nn
import torch.nn.functional as F


def cross_entropy_loss(logits, targets):
    logits  = torch.reshape(logits,[-1,logits.shape[-1]])
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, stats_dict, pred_dict


def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos'):
    '''
    support_embedding shape is n_cont * d * N_support_set where n_cont = n_way * n_shot
    '''

    n_way = len(query_labels.unique())

    prots  = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)  # N_way * D
    embeds = query_embeddings.unsqueeze(1)   # M * 1 * D

    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)

    return cross_entropy_loss(logits, query_labels)


def compute_prototypes(embeddings, labels, n_way):

    '''
        embedding shape is n_cont * [d * N_support_set] where n_cont = n_way * n_shot
        n_way: the number of class of this few-shot task
        prots shape is n_way * [d*N_support_set]
    '''

    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
    return prots


class AdaptiveCosineNCC(nn.Module):
    def __init__(self):
        super(AdaptiveCosineNCC, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(self, support_embeddings, support_labels,
                query_embeddings, query_labels, return_logits=False):
        n_way = len(query_labels.unique())

        prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
        embeds = query_embeddings.unsqueeze(1)
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * self.scale

        if return_logits:
            return logits

        return cross_entropy_loss(logits, query_labels)

