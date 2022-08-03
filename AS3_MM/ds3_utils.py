import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.special import softmax
from DS3 import DS3


def make_onehot_vec(a):
    a = a.astype(np.int)
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def _get_margin(probs):
    ret = np.zeros(len(probs))
    for i, row in enumerate(probs):
        tmp = row[row.argsort()[-2:]]
        ret[i] = tmp[1] - tmp[0]
    return ret

def compute_prototype(support_feas,support_labels):
    unique_labels = np.unique(support_labels)
    n_category    = unique_labels.shape[0]
    prots         = np.zeros((n_category,support_feas.shape[1]))

    for i, l in enumerate(unique_labels):
        idx         = np.where(support_labels == l)[0]
        prots[i,:]  = support_feas[idx, :].mean(0)

    return prots

def distance_predict_matching_net(support_feas,support_labels):

    prototypes    = compute_prototype(support_feas,support_labels)
    shot_dist     = cosine_distances(support_feas,prototypes)
    probs         = softmax(-shot_dist,axis=1)
    margins       = []
    preds         = probs.argmax(1)
    for i, row in enumerate(probs):
        if preds[i] != support_labels[i]:

            margins.append(0)

        else:
            tmp = row[row.argsort()[-2:]]
            margins.append(tmp[1])

    margins    = np.array(margins).reshape([1,-1])

    return probs,margins,prototypes

def mat_support_nlog_likelihood(support_feas, support_labels):
    prots     = compute_prototype(support_feas, support_labels)
    shot_dist = euclidean_distances(support_feas, prots)
    probs     = softmax(-shot_dist, axis=1)
    preds     = probs.argmax(1)
    s         = np.mean(np.equal(preds, support_labels))
    return s


def ds3(support_feas_dict,support_labels):

    feas_keys      = list(support_feas_dict.keys())
    support_labels = support_labels
    total_margins  = []
    all_prots      = []
    prior          = []
    for key in feas_keys:
        support_feas        = support_feas_dict[key]
        probs,margins,prots = distance_predict_matching_net(support_feas, support_labels)
        s                   = mat_support_nlog_likelihood(support_feas, support_labels)
        total_margins.append(margins)
        all_prots.append(prots)
        prior.append(1-s)
    prior            = np.array(prior)
    total_margins    = np.vstack(total_margins)
    dissimat         = 1 - total_margins
    l_star           = np.sum(dissimat,axis=1).argmin()
    #########
    p  = 2
    #########
    if p == np.inf:
        model_reg          = np.asarray([np.linalg.norm(dissimat[i] - dissimat[l_star], ord=1) for i in range(len(dissimat))]).max() / 2
        model_reg          = model_reg * 0.9
    elif p == 2:
        l_star = dissimat.sum(axis=1).argmin()
        rho_max_all = np.zeros(len(dissimat))
        for i in range(len(dissimat)):
            v = dissimat[i] - dissimat[l_star]
            if np.sum(v) > 0:
                rho_max_all[i] = np.divide(np.sqrt(len(dissimat)) * np.linalg.norm(v, ord=2) ** 2, 2 * np.sum(v))
        model_reg  = rho_max_all.max()
        model_reg  = model_reg

    learner_weight     = DS3(dissimat, model_reg).ADMM(0.2, 1e-4, 1e3, p, [], 0)
    return learner_weight,all_prots







