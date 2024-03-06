import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.NLLLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()



def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        # pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary
    else:
        return F.binary_cross_entropy_with_logits(pred, true.float(),
                                                  weight=weight[true])


def bcelogits_loss(y_pred, labels, label_index, num_nodes):
    y_pred = y_pred[label_index[0], label_index[1]].flatten()
    return F.binary_cross_entropy_with_logits(y_pred, labels)


def float_bcelogits_loss(y_pred, labels):
    return F.binary_cross_entropy_with_logits(y_pred, labels.type(torch.float32))


def auc_loss(y_pred, labels, label_index, num_nodes):
    pos_edge_index = label_index[:, labels == 1]
    pos_out = y_pred[pos_edge_index[0], pos_edge_index[1]].view(-1, 1)

    num_pos_edges = pos_edge_index.shape[1]
    neg_edge_index = label_index[:, num_pos_edges:]
    neg_out = y_pred[neg_edge_index[0], neg_edge_index[1]].view(num_pos_edges, -1)

    return torch.square(1 - (pos_out - neg_out)).sum() / num_pos_edges