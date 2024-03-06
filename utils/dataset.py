import math
import os
import os.path as osp
import pickle
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset

class PygLinkPropPredDataset(PygLinkPropPredDataset):
    def __init__(self, name, root, transform=None, pre_transform=None, meta_dict = None):
        super(PygLinkPropPredDataset, self).__init__(name, root, transform, pre_transform, meta_dict)
        self.data_path = root
        self.dataset_name = name

    def get_ogb_pos_edges(self, split_edge, split):
        if 'edge' in split_edge[split]:
            pos_edge = split_edge[split]['edge']
        elif 'head' in split_edge[split]:
            pos_edge = torch.stack([split_edge[split]['head'], split_edge[split]['tail']],
                                dim=1)
        else:
            raise NotImplementedError
        return pos_edge

    def get_same_source_negs(self, num_nodes, num_negs_per_pos, pos_edge):
        """
        The ogb-citation datasets uses negatives with the same src, but different dst to the positives
        :param num_nodes: Int node count
        :param num_negs_per_pos: Int
        :param pos_edge: Int Tensor[2, edges]
        :return: Int Tensor[2, edges]
        """
        print(f'generating {num_negs_per_pos} single source negatives for each positive source node')
        dst_neg = torch.randint(0, num_nodes, (1, pos_edge.size(1) * num_negs_per_pos), dtype=torch.long)
        src_neg = pos_edge[0].repeat_interleave(num_negs_per_pos)
        return torch.cat([src_neg.unsqueeze(0), dst_neg], dim=0)

    def make_obg_supervision_edges(self, split_edge, split, neg_edges=None):
        if neg_edges is not None:
            neg_edges = neg_edges
        else:
            if 'edge_neg' in split_edge[split]:
                neg_edges = split_edge[split]['edge_neg']
            elif 'tail_neg' in split_edge[split]:
                n_neg_nodes = split_edge[split]['tail_neg'].shape[1]
                neg_edges = torch.stack([split_edge[split]['head'].unsqueeze(1).repeat(1, n_neg_nodes).ravel(),
                                        split_edge[split]['tail_neg'].ravel()
                                        ]).t()
            else:
                raise NotImplementedError

        pos_edges = self.get_ogb_pos_edges(split_edge, split)
        n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
        edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
        edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()
        return edge_label, edge_label_index

    def get_ogb_train_negs(self, split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
        """
        for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
        @param split_edge:

        @param edge_index: A [2, num_edges] tensor
        @param num_nodes:
        @param num_negs: the number of negatives to sample for each positive
        @return: A [num_edges * num_negs, 2] tensor of negative edges
        """
        pos_edge = self.get_ogb_pos_edges(split_edge, 'train').t()
        if dataset_name is not None and dataset_name.startswith('ogbl-citation'):
            neg_edge = self.get_same_source_negs(num_nodes, num_negs, pos_edge)
        else:  # any source is fine
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1) * num_negs)
        return neg_edge.t()

    def get_ogb_data(self, data, split_edge, num_negs=1):
        """
        ogb datasets come with fixed train-val-test splits and a fixed set of negatives against which to evaluate the test set
        The dataset.data object contains all of the nodes, but only the training edges
        @param dataset:
        @param use_valedges_as_input:
        @return:
        """
        if num_negs == 1:
            dataset_path_name = self.dataset_name.replace('-', '_')
            negs_name = f'{self.data_path}{dataset_path_name}/negative_samples.pt'
        else:
            dataset_path_name = self.dataset_name.replace('-', '_')
            negs_name = f'{self.data_path}{dataset_path_name}/negative_samples_{num_negs}.pt'
        print(f'looking for negative edges at {negs_name}')
        if os.path.exists(negs_name):
            print('loading negatives from disk')
            train_negs = torch.load(negs_name)
        else:
            print('negatives not found on disk. Generating negatives')
            train_negs = self.get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, self.dataset_name)
            torch.save(train_negs, negs_name)
        splits = {}
        for key in split_edge.keys():
            # the ogb datasets come with test and valid negatives, but you have to cook your own train negs
            neg_edges = train_negs if key == 'train' else None
            edge_label, edge_label_index = self.make_obg_supervision_edges(split_edge, key, neg_edges)
            # use the validation edges for message passing at test time
            # according to the rules https://ogb.stanford.edu/docs/leader_rules/ only collab can use val edges at test time
            if key == 'test' and self.dataset_name == 'ogbl-collab':
                vei, vw = to_undirected(split_edge['valid']['edge'].t(), split_edge['valid']['weight'])
                edge_index = torch.cat([data.edge_index, vei], dim=1)
                edge_weight = torch.cat([data.edge_weight, vw.unsqueeze(-1)], dim=0)
            else:
                edge_index = data.edge_index
                if hasattr(data, "edge_weight"):
                    edge_weight = data.edge_weight
                else:
                    edge_weight = torch.ones(data.edge_index.shape[1])
            if 'laplacian_eigenvector_pe' in data.keys:
                splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                                   edge_label_index=edge_label_index, laplacian_eigenvector_pe=data.laplacian_eigenvector_pe)
            elif 'random_walk_pe' in data.keys:
                splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                                edge_label_index=edge_label_index, random_walk_pe=data.random_walk_pe)
            elif 'pos' in data.keys:
                splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                                edge_label_index=edge_label_index, pos=data.pos)
            else:
                splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                                edge_label_index=edge_label_index)
        return splits


class Zinc12KDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Zinc12KDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Zinc.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b = self.processed_paths[0]       
        a = sio.loadmat(self.raw_paths[0]) 
        # list of adjacency matrix
        F = a['F'][0]
        A = a['E'][0]
        Y = a['Y']
        nmax = 37
        ntype = 21
        maxdeg = 4

        data_list = []
        for i in range(len(A)):
            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.zeros(A[i].shape[0],ntype+maxdeg)
            deg = (A[i] > 0).sum(1)
            for j in range(F[i][0].shape[0]):
                # put atom code
                x[j, F[i][0][j]] = 1
                # put degree code
                x[j, -int(deg[j])] = 1
            y = torch.tensor(Y[i, :])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b = self.processed_paths[0]       
        a = sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A = a['A'][0]
        # list of output
        Y = a['F']

        data_list = []
        for i in range(len(A)):
            a = A[i]
            A2 = a.dot(a)
            A3 = A2.dot(a)
            tri = np.trace(A3)/6
            tailed = ((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4 = 1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus = a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg = a.sum(0)
            star = 0
            for j in range(a.shape[0]):
                star += math.comb(int(deg[j]),3)

            expy = torch.tensor([[tri, tailed, star, cyc4, cus]])

            E = np.where(A[i]>0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0],1)
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Grapg8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Grapg8cDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph8c.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for _, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(Data.from_dict(data)) for data in data_list]
        else:
            data_list = [Data.from_dict(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SpectralDesign(object):   
    def __init__(self, nmax=0, recfield=1, dv=5, nfreq=5, adddegree=False, laplacien=True, addadj=False, vmax=None):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area 
        self.recfield = recfield  
        # b parameter
        self.dv = dv
        # number of sampled point of spectrum
        self.nfreq = nfreq
        # if degree is added to node feature
        self.adddegree = adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien = laplacien
        # add adjacecny as edge feature
        self.addadj = addadj
        # use given max eigenvalue
        self.vmax = vmax

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax = nmax    

    def __call__(self, data):

        n = data.x.shape[0]
        data.x = data.x.type(torch.float32)
               
        nsup = self.nfreq + 1
        if self.addadj:
            nsup += 1
            
        A = np.zeros((n, n), dtype = np.float32)
        SP = np.zeros((nsup,n,n),dtype=np.float32) 
        A[data.edge_index[0], data.edge_index[1]] = 1
        
        if self.adddegree:
            data.x = torch.cat([data.x, torch.tensor(A.sum(0)).unsqueeze(-1)], 1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield == 0:
            M = A
        else:
            M = (A + np.eye(n))
            for i in range(1, self.recfield):
                M = M.dot(M) 
        M = (M > 0)
        d = A.sum(axis=0) 

        # normalized Laplacian matrix.
        dis = 1 / np.sqrt(d)
        dis[np.isinf(dis)] = 0
        dis[np.isnan(dis)] = 0
        D = np.diag(dis)
        nL = np.eye(D.shape[0]) - (A.dot(D)).T.dot(D)
        V, U = np.linalg.eigh(nL) 
        V[V < 0] = 0

        if not self.laplacien:        
            V, U = np.linalg.eigh(A)

        # design convolution supports
        vmax = self.vmax
        if vmax is None:
            vmax = V.max()

        freqcenter = np.linspace(V.min(), vmax, self.nfreq)
        
        # design convolution supports (aka edge features)         
        for i in range(0, len(freqcenter)): 
            SP[i, :, :] = M * (U.dot(np.diag(np.exp(-(self.dv * (V - freqcenter[i])**2))).dot(U.T))) 
        # add identity
        SP[len(freqcenter), :, :] = np.eye(n)
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter)+1, :, :] = A
           
        # set convolution support weigths as an edge feature
        E = np.where(M > 0)
        data.edge_index2 = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:, E[0], E[1]].T).type(torch.float32)        

        return data


class YandexDataset:
    def __init__(self, name, fold_idx, root_path, device='cpu', transform=None):

        print('Preparing data...')
        data = np.load(root_path + f'{name.replace("-", "_")}.npz')
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges']).T
        if name not in ['squirrel-directed', 'squirrel-filtered-directed', 'chameleon-directed', 'chameleon-filtered-directed']:
            edges = to_undirected(edges)

        num_classes = len(labels.unique())
        num_targets = num_classes

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        self.graph = Data(node_features, edges, y=labels)
        if transform is not None:
            self.graph = transform(self.graph)
        self.graph.train_mask = train_masks[fold_idx]
        self.graph.val_mask = val_masks[fold_idx]
        self.graph.test_mask = test_masks[fold_idx]

        self.name = name
        self.device = device

        self.num_targets = num_targets


    # def compute_metrics(self, logits):
    #     if self.num_targets == 1:
    #         train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
    #                                      y_score=logits[self.train_idx].cpu().numpy()).item()

    #         val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
    #                                    y_score=logits[self.val_idx].cpu().numpy()).item()

    #         test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
    #                                     y_score=logits[self.test_idx].cpu().numpy()).item()

    #     else:
    #         preds = logits.argmax(axis=1)
    #         train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
    #         val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
    #         test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

    #     metrics = {
    #         f'train {self.metric}': train_metric,
    #         f'val {self.metric}': val_metric,
    #         f'test {self.metric}': test_metric
    #     }

    #     return metrics
