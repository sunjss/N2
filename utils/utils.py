import csv
import os

import gdown
import numpy as np
import pandas as pd
import torch
import torch_geometric.datasets as GeoData
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.loader import (ClusterData, ClusterLoader, DataListLoader, DataLoader, 
                                    GraphSAINTNodeSampler, RandomNodeSampler)
from torch_geometric.utils import homophily, index_to_mask, degree
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import StratifiedKFold

from utils.transform import (IrrgularFeatIdx2OneHotPre)
from utils.dataset import YandexDataset
from utils.ncd_dataset import load_nc_dataset
from utils.evaluators import (accuracy, eval_rocauc, eval_average_precision)
from os import path

root = path.dirname(path.abspath(__file__))[:-6] + '/data/'
splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}

dataset_list = ["COLLAB", "IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "NCI1", 
                "arxiv-year", "genius",  
                'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                "AmazonComputers", "AmazonPhoto", "CoauthorCS", "CoauthorPhysics",
                "ogbn-arxiv", "ogbn-proteins", "ogbg-molpcba"]

class N2DataLoader:
    def __init__(self, sampler='random'):
        self.sampler = sampler
        self.saint_batch_size = 1000

    def load_data(self, dataset='CORA', spilit_type="public", nbatch=1, fold_idx=0):
        if dataset in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]:
            self.load_tu(dataset, nbatch, fold_idx, True)
        elif dataset in ["PROTEINS", "ENZYMES", "NCI1"]:
            self.load_tu(dataset, nbatch, fold_idx, False)
        elif "ogbg" in dataset:
            self.load_ogbg(dataset, nbatch)
        elif "ogbn" in dataset:
            self.load_ogbn(dataset, nbatch)
        elif dataset in ["AmazonComputers", "AmazonPhoto"]:
            self.load_Amazon(dataset[6:], nbatch)
        elif dataset in ["CoauthorCS", "CoauthorPhysics"]:
            self.load_Coauthor(dataset[8:], nbatch)
        elif dataset in ["arxiv-year", "genius"]:
            self.load_nc(dataset, nbatch, fold_idx)
        elif dataset in ['amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
            self.load_yandex(dataset, nbatch, fold_idx)
        else:
            sup_dataset = "\nSupported datasets include: "
            for i in dataset_list:
                sup_dataset += i
                sup_dataset += ", "
            sup_dataset = sup_dataset[:-2]
            raise ValueError("Unsupported type: " + dataset + sup_dataset)
    
    
    def generate_splits(self, data, g_split):
        n_nodes = len(data.x)
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        idx = torch.randperm(n_nodes)
        val_num = test_num = int(n_nodes * (1 - g_split) / 2)
        train_mask[idx[val_num + test_num:]] = True
        valid_mask[idx[:val_num]] = True
        test_mask[idx[val_num:val_num + test_num]] = True
        data.train_mask = train_mask
        data.val_mask = valid_mask
        data.test_mask = test_mask
        return data


    def load_Amazon(self, dataset, batch_size):
        data_path = root + "amazon/"
        pre_transform = T.Compose([]) #FeatIdx2OneHotPre(21), EdgeIdx2OneHotPre(4), T.RootedRWSubgraph(32) T.RootedEgoNets(3) T.AddLaplacianEigenvectorPE(6)
        dataset = GeoData.Amazon(data_path, name=dataset, pre_transform=pre_transform)
        data = dataset[0]
        data = self.generate_splits(data, 0.6)
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.test_data = DataListLoader(dataset=[data], batch_size=1)
        else:
            self.train_data = DataLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=[data], batch_size=batch_size)
            self.test_data = DataLoader(dataset=[data], batch_size=batch_size)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features
        self.task_type = "single-class"
        self.metric = accuracy 
    

    def load_Coauthor(self, dataset, batch_size):
        data_path = root + "coauthor/"
        pre_transform = T.Compose([]) #FeatIdx2OneHotPre(21), EdgeIdx2OneHotPre(4), T.RootedRWSubgraph(32) T.RootedEgoNets(3) T.AddLaplacianEigenvectorPE(6)
        dataset = GeoData.Coauthor(data_path, name=dataset, pre_transform=pre_transform)
        data = dataset[0]
        data = self.generate_splits(data, 0.6)
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.test_data = DataListLoader(dataset=[data], batch_size=1)
        else:
            self.train_data = DataLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=[data], batch_size=batch_size)
            self.test_data = DataLoader(dataset=[data], batch_size=batch_size)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features
        self.task_type = "single-class"
        self.metric = accuracy 


    def load_yandex(self, dataset_type, batch_size, fold_idx):
        data_path = root + 'yandex/'
        if dataset_type in ["squirrel", "chameleon"]:
            print(dataset_type.title() + " from Yandex")
        dataset = YandexDataset(dataset_type, fold_idx, data_path)#, transform=T.AddLaplacianEigenvectorPE(6))
        data = dataset.graph
        if batch_size > 1:
            if self.sampler == "random":
                self.train_data = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "graphsaint":
                self.train_data = GraphSAINTNodeSampler(data, batch_size=self.saint_batch_size, num_steps=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "cluster":
                cluster_data = ClusterData(data, num_parts=batch_size, save_dir=data_path+'cluster/'+dataset_type)
                self.train_data = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
                # graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
                graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
            else:
                raise ValueError("Unsupported type: " + self.sampler)
            self.val_data = graph_loader
            self.test_data = graph_loader
        else:
            self.train_data = DataLoader([data], batch_size=batch_size, shuffle=True, num_workers=0)
            self.val_data = DataLoader([data], batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader([data], batch_size=batch_size, shuffle=False)
        self.nclass = dataset.num_targets
        self.nfeats = data.num_features
        self.nedgefeats = data.num_edge_features
        self.task_type = "single-class"
        self.ndits = 6
        if dataset_type in ['minesweeper', 'tolokers', 'questions']:
            self.metric = eval_rocauc
        else:
            self.metric = accuracy


    def load_fixed_splits(self, dataset, sub_dataset):
        """ loads saved fixed splits for dataset
        """
        name = dataset
        if sub_dataset and sub_dataset != 'None':
            name += f'-{sub_dataset}'

        if not os.path.exists(root + f'nc/splits/{name}-splits.npy'):
            assert dataset in splits_drive_url.keys()
            gdown.download(id=splits_drive_url[dataset], \
                           output=root + f'nc/splits/{name}-splits.npy', quiet=False) 
        
        splits_lst = np.load(root + f'nc/splits/{name}-splits.npy', allow_pickle=True)
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        return splits_lst

    def load_nc(self, dataset_type, batch_size, fold_idx):
        data, sub_dataname = load_nc_dataset(dataset_type)
        data = Data(data.graph["node_feat"], data.graph["edge_index"], data.graph["edge_feat"], data.label)
        
        split_idx_lst = self.load_fixed_splits(dataset_type, sub_dataname)
        data.train_mask = index_to_mask(split_idx_lst[fold_idx]['train'], data.num_nodes)
        data.val_mask = index_to_mask(split_idx_lst[fold_idx]['valid'], data.num_nodes)
        data.test_mask = index_to_mask(split_idx_lst[fold_idx]['test'], data.num_nodes)
        if data.y.min() == -1:
            data.mask = torch.ne(data.y, -1)
        else:
            data.mask = None

        if batch_size > 1:
            if self.sampler == "random":
                self.train_data = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "graphsaint":
                self.train_data = GraphSAINTNodeSampler(data, batch_size=self.saint_batch_size, num_steps=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "cluster":
                cluster_data = ClusterData(data, num_parts=batch_size, save_dir=root+'nc/cluster/'+dataset_type)
                self.train_data = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
                graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
            else:
                raise ValueError("Unsupported type: " + self.sampler)
            self.val_data = graph_loader
            self.test_data = graph_loader
        else:
            self.train_data = DataLoader([data], batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader([data], batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader([data], batch_size=batch_size, shuffle=False)
        self.nclass = data.y.max().item() + 1
        self.nfeats = data.num_features
        self.nedgefeats = data.num_edge_features
        self.task_type = "single-class"
        if dataset_type in ['genius']:
            self.metric = eval_rocauc
        else:
            self.metric = accuracy

    def random_disassortative_splits(self, labels):
        # * 0.6 labels for training
        # * 0.2 labels for validation
        # * 0.2 labels for testing
        num_classes = labels.max() + 1
        num_classes = num_classes.numpy()
        indices = []
        for i in range(num_classes):
            index = torch.nonzero((labels == i)).view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        percls_trn = int(round(0.6 * (labels.size()[0] / num_classes)))
        val_lb = int(round(0.2 * labels.size()[0]))
        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=labels.size()[0])
        val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
        test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

        return train_mask, val_mask, test_mask


    def load_ogbg(self, dataset_type, batch_size):
        data_path = root + 'ogb/'
        dataset = PygGraphPropPredDataset(name=dataset_type, root=data_path)
        
        split_idx = dataset.get_idx_split() 
        
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=dataset[split_idx["valid"]], batch_size=batch_size)
            self.test_data = DataListLoader(dataset=dataset[split_idx["test"]], batch_size=1)
        else:
            self.train_data = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features
        self.task_type = "single-class"
        if dataset_type == "ogbg-molpcba":
            self.metric = eval_average_precision
            self.task_type = "multi-class"
            self.nclass = dataset.num_tasks
        elif dataset_type == "ogbg-molhiv":
            self.metric = eval_rocauc
            self.task_type = "binary-class"
            self.nclass = 1


    def __set_dataset_attr(self, dataset, name, value, size):
        dataset._data_list = None
        dataset.data[name] = value
        if dataset.slices is not None:
            dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


    def load_ogbn(self, dataset_type, batch_size):
        data_path = root + 'ogb/'
        if dataset_type == "ogbn-proteins":
            pre_transform = T.Compose([IrrgularFeatIdx2OneHotPre()])
        else:
            pre_transform = T.Compose([]) #, T.RootedRWSubgraph(32) T.RootedEgoNets(3) T.AddLaplacianEigenvectorPE(6)
        dataset = PygNodePropPredDataset(name=dataset_type, root=data_path, pre_transform=pre_transform)
        
        split_idx = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i, key in enumerate(split_idx.keys()):
            mask = index_to_mask(split_idx[key], dataset.data.num_nodes)
            self.__set_dataset_attr(dataset, split_names[i], mask, dataset.data.num_nodes)
        if batch_size > 1:
            data = dataset[0]
            if self.sampler == "random":
                self.train_data = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "graphsaint":
                self.train_data = GraphSAINTNodeSampler(data, batch_size=self.saint_batch_size, num_steps=5, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "cluster":
                cluster_data = ClusterData(data, num_parts=batch_size, save_dir=data_path+'cluster/'+dataset_type)
                self.train_data = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
                graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
            else:
                raise ValueError("Unsupported type: " + self.sampler)
            self.val_data = graph_loader
            self.test_data = graph_loader
        else:
            self.train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features
        self.metric = accuracy
        self.task_type = "single-class"
        if dataset_type == "ogbn-proteins":
            self.task_type = "multi-class"
            self.metric = eval_rocauc
            self.nclass = dataset.num_tasks


    def csv_reader(self, file, fold_idx):
        with open(file, "rt") as f:
            data = pd.read_csv(f, delimiter='\t', header=None)
            idx = data.values[fold_idx][0]
        idx = idx.split(',')
        idx = torch.tensor(list(map(int, idx)))
        return idx


    def csv_writer(self, labels, path, seed=42):
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
        train_idx_list = []
        val_idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            train_idx_list.append(idx[0])
            val_idx_list.append(idx[1])

        with open(path + "/train_idx.csv", "wt") as f:
            cw = csv.writer(f)
            cw.writerows(train_idx_list)
        with open(path + "/val_idx.csv", "wt") as f:
            cw = csv.writer(f)
            cw.writerows(val_idx_list)


    def load_tu(self, dataset_type, batch_size, fold_idx, attr=False):
        data_path = root + 'tu/'
        dataset = GeoData.TUDataset(root=data_path, name=dataset_type)
        if attr:
            max_degree = torch.max(degree(dataset.edge_index[0]))
            transform = T.Compose([T.OneHotDegree(int(max_degree.item()))])
            dataset.transform = transform

        flag = os.path.exists(data_path + dataset_type + "/train_idx.csv") and os.path.exists(data_path + dataset_type + "/val_idx.csv")
        if not flag:
            self.csv_writer(dataset.data.y.numpy(), data_path + dataset_type)
        
        train_idx = self.csv_reader(data_path + dataset_type + "/train_idx.csv", fold_idx)
        val_idx = self.csv_reader(data_path + dataset_type + "/val_idx.csv", fold_idx)
        train_dataset = dataset.index_select(train_idx)
        val_dataset = dataset.index_select(val_idx)
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=val_dataset, batch_size=batch_size)
            self.test_data = DataListLoader(dataset=dataset, batch_size=1)
        else:
            self.train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=val_dataset, batch_size=batch_size)
            self.test_data = DataLoader(dataset=dataset, batch_size=batch_size)
        self.nclass = train_dataset.num_classes
        self.nfeats = train_dataset.num_node_features
        self.nedgefeats = train_dataset.num_edge_features
        self.metric = accuracy
        self.task_type = "single-class"



def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        if not p.requires_grad:
            continue
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def output_hetero(perm, output, labels):
    preds = output.max(1)[1].type_as(labels)
    h = []
    for i in range(perm.shape[0]):
        index = perm[i, :, :].to_sparse().indices()
        h.append(homophily(index, preds))
    print(h)
