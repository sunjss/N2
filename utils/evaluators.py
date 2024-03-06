import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, average_precision_score
import torch
import torch.nn.functional as F
from torchmetrics.functional import average_precision
from torch_geometric.utils import to_dense_adj

import logging
import warnings


def eval_average_precision(output, labels):
    warnings.filterwarnings("error")
    if type(output) is list:
        output = torch.cat(output, 0)
    if type(labels) is list:
        labels = torch.cat(labels, 0)
    output = output.detach().cpu().numpy()
    if labels.device != "cpu":
        labels = labels.detach().cpu().numpy()
    return average_precision_score(labels, output)


def macro_f1_score(output, labels):
    logging.getLogger().setLevel(logging.ERROR)
    if type(output) is list:
        output = torch.cat(output, 0)
    if type(labels) is list:
        labels = torch.cat(labels, 0)
    output = torch.argmax(output, -1).detach().cpu().numpy()
    if labels.device != "cpu":
        labels = labels.detach().cpu().numpy()
    print(f1_score(output, labels, average=None, zero_division=0))
    return f1_score(output, labels, average="macro", zero_division=0)


def micro_f1_score(output, labels):
    logging.getLogger().setLevel(logging.ERROR)
    if type(output) is list:
        output = torch.cat(output, 0)
    if type(labels) is list:
        labels = torch.cat(labels, 0)
    output = torch.argmax(output, -1).detach().cpu().numpy()
    if labels.device != "cpu":
        labels = labels.detach().cpu().numpy()
    print(f1_score(output, labels, average=None, zero_division=0))
    return f1_score(output, labels, average="micro", zero_division=0)


def eval_mrr(y_pred_pos, y_pred_neg):
    y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1)
    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    results = 1./ranking_list.to(torch.float).mean().item()
    return results


def eval_hits(y_pred, label, label_index, num_nodes, Ks=[20, 50, 100]):
    y_pred = y_pred[label_index[0], label_index[1]]
    y_pred_pos = y_pred[label == 1]
    y_pred_neg = y_pred[label == 0]
    results = {}
    for K in Ks:
        if len(y_pred_neg) < K:
            return {'hits@{}'.format(K): 1.}
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        results[f'Hits@{K}'] = torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu() / len(y_pred_pos)
    return results


def eval_rocauc(y_pred, y_true):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    y_pred = torch.concat(y_pred, 0)
    if type(y_true) is not torch.Tensor:
        y_true = torch.concat(y_true, 0)
    rocauc_list = []
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(-1)
    y_true = y_true.detach().cpu().numpy()
    if y_true.max() == 1 and y_pred.shape[1] == 2:
        # use the predicted class for single-class classification
        y_pred = y_pred[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


import matplotlib.pyplot as plt
def accuracy_degree(output, labels, data, mask):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])
    deg = adj.sum(-1).view(-1)[mask]
    num = int(deg.max())
    de = torch.histc(deg, bins=num, min=1, max=num)
    nu = torch.histc(deg * correct, bins=num, min=1, max=num).cpu()
    de = torch.where(de==0, torch.ones_like(de), de).cpu()
    plt.bar(torch.arange(0, num), nu / de)
    plt.savefig("res.pdf")


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_SBM(scores, targets):
    '''
    The performance measure is the average node-level accuracy weighted with respect to the class sizes
    '''
    S = targets.cpu().numpy()
    C = np.argmax(scores.cpu().detach().numpy() , axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc


def eval_mrr_and_hits(y_pred, label, label_index, num_nodes):
    """ Compute Hits@k and Mean Reciprocal Rank (MRR).

    Implementation from OGB:
    https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

    Args:
        y_pred_neg: array with shape (batch size, num_entities_neg).
        y_pred_pos: array with shape (batch size, )
    """
    pos_edge_index = label_index[:, label == 1]
    y_pred_pos = y_pred[pos_edge_index[0], pos_edge_index[1]]

    num_pos_edges = pos_edge_index.shape[1]
    neg_mask = torch.ones([num_pos_edges, num_nodes], dtype=torch.bool)
    neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
    y_pred_neg = y_pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)

    y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
    argsort = torch.argsort(y_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)
    return {'hits@1': hits1_list.sum() / hits1_list.numel(),
            'hits@3': hits3_list.sum() / hits3_list.numel(),
            'hits@10': hits10_list.sum() / hits10_list.numel(),
            'mrr': mrr_list.sum() / mrr_list.numel()}
    # y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg], axis=1)
    # argsort = np.argsort(-y_pred, axis=1)
    # ranking_list = (argsort == 0).nonzero()
    # ranking_list = ranking_list[1] + 1
    # hits1_list = (ranking_list <= 1).astype(np.float32)
    # hits3_list = (ranking_list <= 3).astype(np.float32)
    # hits10_list = (ranking_list <= 10).astype(np.float32)
    # mrr_list = 1. / ranking_list.astype(np.float32)

    # return {'hits@1_list': hits1_list,
    #         'hits@3_list': hits3_list,
    #         'hits@10_list': hits10_list,
    #         'mrr_list': mrr_list}