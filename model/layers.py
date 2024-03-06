import torch
import torch.nn as nn
from torch_geometric.utils import to_torch_coo_tensor


class PathIntegral(nn.Module):
    def __init__(self, q_dim, n_q):
        super(PathIntegral, self).__init__()
        self.lambda_copies = nn.Parameter(torch.randn(n_q, 1, 1))
        self.n_q = n_q
        self.q_dim = q_dim
        self.in_subsystem = None
        self.out_subsystem = None


    def _path_integral(self, in_subsystem=None, out_subsystem=None):
        if in_subsystem == None:
            in_subsystem = self.in_subsystem
        if out_subsystem == None:
            out_subsystem = self.out_subsystem
        if in_subsystem == None or out_subsystem == None:
            raise ValueError("Path integral requires computational object.")
        
        # TODO: ablation study between attention and dist
        if in_subsystem.shape[-3] == 1 and out_subsystem.shape[-3] == self.n_q:
            out_subsystem = out_subsystem * self.lambda_copies
            out_subsystem_sum = out_subsystem.sum(-3) / (self.n_q * self.q_dim)
            weighted_dist_sum = torch.matmul(in_subsystem.squeeze(-3), 
                                             out_subsystem_sum.transpose(-2, -1))
        elif in_subsystem.shape[-3] == self.n_q and out_subsystem.shape[-3] == 1:
            in_subsystem = in_subsystem * self.lambda_copies
            in_subsystem_sum = in_subsystem.sum(-3) / (self.n_q * self.q_dim)
            weighted_dist_sum = torch.matmul(in_subsystem_sum, 
                                             out_subsystem.squeeze(-3).transpose(-2, -1))
        else:
            dist = torch.matmul(in_subsystem, out_subsystem.transpose(-2, -1)) / self.q_dim
            # dist = torch.tanh(dist)
            weighted_dist = dist * self.lambda_copies
            weighted_dist_sum = weighted_dist.sum(-3) / self.n_q  # (n_in, n_out)

        # dist clip
        # weighted_dist_sum = torch.tanh(weighted_dist_sum)
        with torch.no_grad():
            clip_check = torch.abs(weighted_dist_sum.sum(-1, keepdim=True))
            fill_value = torch.ones_like(clip_check)
            scaler = torch.where(clip_check > 1e+4, 1e+4 / clip_check, fill_value)
        weighted_dist_sum = scaler * weighted_dist_sum
        return weighted_dist_sum


    def forward(self, in_subsystem, out_subsystem):
        return self._path_integral(in_subsystem, out_subsystem)



class PNodeCommunicator(nn.Module):
    def __init__(self, d_in, d_out, q_dim, n_q, dropout):
        super(PNodeCommunicator, self).__init__()
        self.q_dim = q_dim
        self.pnode_agg = PathIntegral(q_dim, n_q)
        self.glob2disp = nn.Sequential(nn.Linear(d_in, q_dim * n_q), 
                                        nn.LeakyReLU(), 
                                        nn.Dropout(dropout))
        self.glob2value = nn.Sequential(nn.Linear(d_in, d_out), 
                                        nn.LeakyReLU(), 
                                        nn.Dropout(dropout))


    def forward(self, state, glob):
        glob_updater = self.pnode_agg(state, state)  # (n_pnode, n_pnode)
        glob_update = torch.matmul(glob_updater, glob)  # (n_pnode, d_in)

        displacement = self.glob2disp(glob_update)  # (n_pnode, q_dim)
        displacement = displacement.unflatten(-1, (self.q_dim, -1))
        displacement = displacement.permute(0, 3, 1, 2)

        dispatch_value = self.glob2value(glob_update)  # (n_pnode, d_out)

        return displacement, dispatch_value


class NodePseudoSubsystem(nn.Module):
    '''
    Neurons as Nodes: get nodes' neuronal state through neurons as nodes
    '''
    def __init__(self, d_in, d_ein, d_out, n_pnode, d_model, q_dim, n_q, dropout=0.0, norm=True):
        super(NodePseudoSubsystem, self).__init__()
        self.collection1 = PathIntegral(q_dim, n_q)
        self.pnode_agg1 = PNodeCommunicator(d_model, d_model, q_dim, n_q, dropout)

        self.inspection = PathIntegral(q_dim, n_q)
        self.edge_wise_ff = nn.Linear(d_model, d_model)
        self.hstate_interface = nn.Sequential(nn.Linear(d_model * 2 + q_dim, q_dim), 
                                              nn.LeakyReLU(), 
                                              nn.Dropout(dropout))

        self.collection2 = PathIntegral(q_dim, n_q)
        self.pnode_agg2 = PNodeCommunicator(d_model * 3 + q_dim, d_out, q_dim, n_q, dropout)
        
        self.dispatch = PathIntegral(q_dim, n_q)
        self.feat_ff = nn.Sequential(nn.Linear(q_dim, d_model), 
                                     nn.LeakyReLU(), 
                                     nn.Dropout(dropout))
        if norm:
            self.phidden_norm = nn.LayerNorm(q_dim)
            self.hidden_norm = nn.LayerNorm(q_dim)
            self.pout_norm = nn.LayerNorm(q_dim)
            self.out_norm = nn.LayerNorm(q_dim)
            self.feat_norm = nn.LayerNorm(d_model)
        self.norm = norm
        print(f"Using norm: {norm}" )

        self.time_embedding = nn.Embedding(6, d_model)
        self.q_dim = q_dim
        self.pnode_num = n_pnode
        self.d_model = d_model


    def _feature_inspection(self, features, node_state, pnode_state, node_num):
        # init feature inspection (node to pnode, pnode-level learning)
        ipn2n_dist = self.collection1(pnode_state, node_state)  # (n_pnode, n)
        glob_init = torch.matmul(ipn2n_dist, features) / node_num  # (n_pnode, d_model)
        pnode_disp1, self.str_inspector = self.pnode_agg1(pnode_state, glob_init)
        pnode_state = pnode_disp1 + pnode_state
        if self.norm:
            pnode_state = self.phidden_norm(pnode_state)

        # inspector dispatch (pnode to node, node-level learning)
        self.pnode_state = pnode_state
        n2ipn_dist = self.inspection(node_state, pnode_state)  # (n, n_pnode)
        inspector = torch.matmul(n2ipn_dist, self.str_inspector)  # (n, d_model)
        return inspector, pnode_state


    def _pnode_aggregator(self, pnode_state, hnode_state, insp_out, node_num):
        # feature collection (node to pnode)
        opn2n_dist = self.collection2(pnode_state, hnode_state)  # (n_pnode, n)
        glob_info = torch.matmul(opn2n_dist, insp_out) / node_num  # (n_pnode, d_model * 2)
        glob_info = torch.concat((glob_info, self.str_inspector), -1)

        # pnode-level feature refinement (pnode-level learning)
        pnode_disp2, dispatch_value = self.pnode_agg2(pnode_state, glob_info)  # (n_pnode, n_pnode)
        pnode_state = pnode_state + pnode_disp2
        if self.norm:
            pnode_state = self.pout_norm(pnode_state)
        
        n2opn_dist = self.dispatch(hnode_state, pnode_state)  # (b_s, n, n_pnode)
        dispatch_value = torch.matmul(n2opn_dist, dispatch_value)
        return dispatch_value, pnode_state


    def _edge_aggregation(self, insp_in, edge_index, edge_weight, edge_attr, size):
        adj = to_torch_coo_tensor(edge_index, edge_weight, size=size)
        insp_out = torch.matmul(adj, insp_in)  # (b_s * n, 2 * d_model)
        return insp_out


    def forward(self, *, edge_index=None, 
                         edge_weight=None, 
                         edge_attr=None,
                         features=None, 
                         node_state=None, 
                         pnode_state=None, 
                         mask=None,
                         size=None):
        # node_state (b_s, n, q_dim), features (b_s, n, d_model)
        b_s, n = features.shape[:2]
        node_num = n if mask is None else mask.sum(-2, keepdim=True)
        
        # inspector generation
        insp, pnode_state = self._feature_inspection(features, 
                                                     node_state.unsqueeze(1),
                                                     pnode_state,
                                                     node_num)
        
        # feature inspection
        insp_in = torch.concat((features, insp, node_state), -1)
        insp_out = self._edge_aggregation(insp_in.flatten(0, -2),
                                          edge_index,
                                          edge_weight, 
                                          edge_attr, 
                                          size).view(b_s, n, -1)
        hnode_state = self.hstate_interface(insp_out)  # (b_s, n, q_dim)
        hnode_state = hnode_state + node_state
        if self.norm:
            hnode_state = self.hidden_norm(hnode_state)

        if mask is not None:
            hnode_state = mask * hnode_state
            
        # feature aggregation
        dispatch_value, pnode_state = self._pnode_aggregator(pnode_state, 
                                                             hnode_state.unsqueeze(1),
                                                             insp_out,
                                                             node_num)
        update_features = self.feat_ff(dispatch_value)
        features = update_features + features
        node_state = hnode_state + dispatch_value # (n, q_dim)
        if self.norm:
            node_state = self.out_norm(node_state)
            features = self.feat_norm(features)
        if mask is not None:
            node_state = mask * node_state
            features = mask * features

        return node_state, pnode_state, features
