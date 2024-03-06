import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import (to_dense_batch, 
                                   add_remaining_self_loops, 
                                   scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes

from model.layers import PathIntegral
from model.layers import NodePseudoSubsystem

class N2(nn.Module):
    def __init__(self, *, d_in=1, 
                          d_ein=0, 
                          nclass=1, 
                          d_model=64, 
                          q_dim=64, 
                          n_q=8, 
                          n_c=8, 
                          n_pnode=256, 
                          T=1, 
                          task_type="single-class",
                          self_loop=True,
                          pre_encoder=None, 
                          pos_encoder=None, 
                          dropout=0.1):
        super(N2, self).__init__()
        d_in = d_in + d_ein
        self.node_state_interface = nn.Sequential(nn.Linear(d_in, q_dim),
                                                  nn.LeakyReLU(),
                                                  nn.Dropout(dropout))
        self.feat_ff = nn.Sequential(nn.Linear(d_in, d_model),
                                     nn.LeakyReLU(),
                                     nn.Dropout(dropout))
        self.pnode_state = nn.Parameter(torch.randn(1, n_pnode, q_dim))
        if task_type != "reg":
            self.node_state_updater = NodePseudoSubsystem(d_in, d_ein, q_dim, n_pnode, d_model, q_dim, n_q, dropout)
            self.class_neuron = nn.Parameter(torch.randn(n_c, nclass, q_dim))
            self.out_ff = PathIntegral(q_dim, n_q)
        else:
            self.node_state_updater = NodePseudoSubsystem(d_in, d_ein, q_dim, n_pnode, d_model, q_dim, n_q, dropout, False)
        self.pre_encoder = pre_encoder 
        self.pos_encoder = pos_encoder 
        self.task_type = task_type  
        self.T = T
        self.n_q = n_q
        self.q_dim = q_dim
        self.n_pnode = n_pnode
        self.d_model = d_model
        self.self_loop = self_loop
    

    def _get_sparse_normalized_adj(self, *, edge_index=None, max_num_nodes=None, edge_weight=None, batch=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        if edge_weight.dtype == torch.long:
            edge_weight = edge_weight.type(torch.float32)

        # normalize edge weight
        row, col = edge_index[0], edge_index[1]
        deg = scatter(edge_weight, row, 0, 
                      dim_size=maybe_num_nodes(edge_index), 
                      reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] 

        # batch gen
        if batch is None:
            num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
            batch = edge_index.new_zeros(num_nodes)
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

        # transform edge index into batched index with padding
        one = batch.new_ones(batch.size(0))
        num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

        idx0 = batch[edge_index[0]]
        idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
        idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

        if ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
            or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
            mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
            idx0 = idx0[mask]
            idx1 = idx1[mask]
            idx2 = idx2[mask]
            edge_weight = edge_weight[mask]
        
        idx = torch.stack((idx1, idx2), 0)
        idx = idx0 * max_num_nodes + idx
        return idx, edge_weight


    def _feature_prep(self, data):
        features = data.x
        mask = None
        whole_size = features.shape[-2]
        if features.dtype == torch.long:
            features = features.type(torch.float32)

        if self.pre_encoder is not None:
            features = self.pre_encoder(features)
        if self.pos_encoder is not None:
            # features = self.pos_encoder(features, data.EigVals, data.EigVecs)
            features = self.pos_encoder(features, data.rrwp)
        if isinstance(data, Batch):
            features, mask = to_dense_batch(features, data.batch)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        b_s, n = features.shape[:2]
        edge_index, edge_attr = add_remaining_self_loops(data.edge_index, 
                                                         data.edge_attr, 
                                                         num_nodes=whole_size)
        if edge_attr is not None:
            edge_attr = scatter(edge_attr, edge_index[0], 0, 
                                dim_size=maybe_num_nodes(edge_index), 
                                reduce='sum')
            if isinstance(data, Batch):
                edge_attr, _ = to_dense_batch(edge_attr, data.batch)
            edge_attr = edge_attr.view(b_s, n, -1)
            features = torch.concat((features, edge_attr), -1)
            edge_attr = None
        edge_index, edge_weight = self._get_sparse_normalized_adj(edge_index=edge_index, 
                                                                  max_num_nodes=n,
                                                                  batch=data.batch)

        return features, edge_index, edge_weight, edge_attr, mask


    def get_output(self, features):
        if "single-class" in self.task_type:
            output = F.log_softmax(features, dim=-1)
        elif self.task_type in ["multi-class",  "reg"]:
            output = features
        elif self.task_type == "binary-class":
            output = features.flatten()
        elif "link" in self.task_type:
            outdim = features.shape[-1] // 2
            node_in = features[:, :outdim]
            node_out = features[:, outdim:]
            if "scale-dot" in self.task_type:
                output = torch.matmul(node_in, node_out.T) / outdim
            elif "cosine" in self.task_type:
                norm_in = torch.norm(node_in, dim=-1)
                norm_out = torch.norm(node_out, dim=-1)
                output = torch.matmul(node_in, node_out.T) / (norm_in * norm_out)
            output = output * 2
        else:
            raise ValueError("Unsupported task type " + self.task_type)
        return output


    def forward(self):
        pass


class N2Node(N2):
    def __init__(self, *, d_in=1, 
                          d_ein=0,
                          nclass=1, 
                          d_model=64, 
                          q_dim=64, 
                          n_q=8, 
                          n_c=8, 
                          n_pnode=256, 
                          T=1, 
                          task_type="single-class",
                          pre_encoder=None, 
                          pos_encoder=None, 
                          self_loop=True,
                          dropout=0.1):
        super(N2Node, self).__init__(d_in=d_in, 
                                     d_ein=d_ein,
                                     nclass=nclass, 
                                     d_model=d_model, 
                                     q_dim=q_dim, 
                                     n_q=n_q, 
                                     n_c=n_c, 
                                     n_pnode=n_pnode, 
                                     T=T, 
                                     task_type=task_type,
                                     pre_encoder=pre_encoder, 
                                     pos_encoder=pos_encoder, 
                                     self_loop=self_loop,
                                     dropout=dropout)

    def forward(self, data):
        features, edge_index, edge_weight, edge_attr, mask = self._feature_prep(data)
        size = features.shape[-2] if data.batch is None else (data.batch.max() + 1) * features.shape[-2]
        node_state = self.node_state_interface(features)  # (b_s, n, q_dim)
        features = self.feat_ff(features)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            features = mask * features
            node_state = mask * node_state
        
        pnode_state = self.pnode_state
        for t in range(self.T):
            node_state, pnode_state, features = self.node_state_updater(edge_index=edge_index, 
                                                                        edge_weight=edge_weight,
                                                                        edge_attr=edge_attr,
                                                                        features=features, 
                                                                        node_state=node_state, 
                                                                        pnode_state=pnode_state, 
                                                                        mask=mask, 
                                                                        size=size)
        features = self.out_ff(node_state.unsqueeze(1), 
                               self.class_neuron)
        features = features.flatten(0, 1)
        features = features[mask.flatten()] if mask is not None else features
        return self.get_output(features)


class N2Graph(N2):
    def __init__(self, *, d_in=1, 
                          d_ein=0,
                          nclass=1, 
                          d_model=64, 
                          q_dim=64, 
                          n_q=8, 
                          n_c=8, 
                          n_pnode=256, 
                          T=1, 
                          task_type="single-class",
                          pre_encoder=None, 
                          pos_encoder=None, 
                          self_loop=True,
                          dropout=0.1):
        super(N2Graph, self).__init__(d_in=d_in, 
                                      d_ein=d_ein,
                                      nclass=nclass, 
                                      d_model=d_model, 
                                      q_dim=q_dim, 
                                      n_q=n_q, 
                                      n_c=n_c, 
                                      n_pnode=n_pnode, 
                                      T=T, 
                                      task_type=task_type,
                                      pre_encoder=pre_encoder, 
                                      pos_encoder=pos_encoder, 
                                      self_loop=self_loop,
                                      dropout=dropout)
        self.pnode_agg = nn.Sequential(nn.Linear(n_pnode * n_q, 1),
                                        nn.LeakyReLU(), 
                                        nn.Dropout(dropout))
        self.out_norm = nn.LayerNorm(q_dim)


    def forward(self, data):
        features, edge_index, edge_weight, edge_attr, mask = self._feature_prep(data)
        size = features.shape[-2] if data.batch is None else (data.batch.max() + 1) * features.shape[-2]
        node_state = self.node_state_interface(features)  # (b_s, n, q_dim)
        features = self.feat_ff(features)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            features = mask * features
            node_state = mask * node_state
        
        # feature extracting
        pnode_state = self.pnode_state
        for t in range(self.T):
            node_state, pnode_state, features = self.node_state_updater(edge_index=edge_index, 
                                                                        edge_weight=edge_weight,
                                                                        edge_attr=edge_attr,
                                                                        features=features, 
                                                                        node_state=node_state,
                                                                        pnode_state=pnode_state, 
                                                                        mask=mask, 
                                                                        size=size)
        output_pnode_state = pnode_state.permute(0, 3, 1, 2)
        output_pnode_state = output_pnode_state.flatten(-2, -1)
        pooling_state = self.pnode_agg(output_pnode_state)  # ff
        pooling_state = pooling_state.transpose(-2, -1)
        pooling_state = self.out_norm(pooling_state)
        features = self.out_ff(pooling_state.unsqueeze(1), 
                                self.class_neuron)
        features = features.view(pnode_state.shape[0], -1)
        return self.get_output(features)
