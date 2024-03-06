import torch
from torch_geometric.utils import one_hot, scatter, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.transforms import BaseTransform
import numpy as np


class FeatIdx2OneHotPre(BaseTransform):
    def __init__(self, num_classes=None) -> None:
        self.num_classes = num_classes
    
    def __call__(self, data):
        if data.x is None:
            features = data.node_species
        else:
            features = data.x
        if self.num_classes is None:
            self.num_classes = len(features.unique())
        data.x = one_hot(features.view(-1), num_classes=self.num_classes)
        data.num_features = self.num_classes
        data.num_node_features = self.num_classes
        return data


class IrrgularFeatIdx2OneHotPre(BaseTransform):
    def __init__(self, num_classes=None) -> None:
        self.num_classes = num_classes
    
    def __call__(self, data):
        if data.x is None:
            features = data.node_species
        else:
            features = data.x
        ids = features.unique()
        if self.num_classes is None:
            self.num_classes = len(ids)
        id_dict = dict(list(zip(ids.numpy(), np.arange(self.num_classes))))
        one_hot_encoding = torch.zeros((features.shape[0], self.num_classes))
        for i, u in enumerate(features):
            if id_dict[u.item()] == 4:
                pass
            else:
                one_hot_encoding[i][id_dict[u.item()]] = 1
        data.x = one_hot_encoding
        data.num_features = self.num_classes
        data.num_node_features = self.num_classes
        return data


class EdgeIdx2OneHotPre(BaseTransform):
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
    
    def __call__(self, data):
        data.edge_attr = one_hot(data.edge_attr, num_classes=self.num_classes)
        data.num_edge_features = self.num_classes
        return data


class ConcatCoordAndPixel(BaseTransform):
    def __call__(self, data):
        if 'laplacian_eigenvector_pe' in data.keys:
            pe = data.laplacian_eigenvector_pe
        elif 'random_walk_pe' in data.keys:
            pe = data.random_walk_pe
        elif 'pos' in data.keys:
            pe = data.pos
        data.x = torch.concat((data.x, pe), -1)
        return data


class AddNormDeg(BaseTransform):
    def __call__(self, data):
        edge_weight = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
        row, col = data.edge_index[0], data.edge_index[1]
        deg = scatter(edge_weight, row, 0, dim_size=maybe_num_nodes(data.edge_index), reduce='sum')
        deg = deg / deg.max()
        data.x = torch.concat((data.x, deg.unsqueeze(1)), -1)
        return data


class AddDeg(BaseTransform):
    def __call__(self, data):
        row, col = data.edge_index[0], data.edge_index[1]
        deg = degree(row)
        if data.x is None:
            data.x = deg.unsqueeze(1)
        else:
            data.x = torch.concat((data.x, deg.unsqueeze(1)), -1)
        return data


class AddOneHotDeg(BaseTransform):
    def __call__(self, data):
        row, col = data.edge_index[0], data.edge_index[1]
        deg = degree(row)
        deg = one_hot(deg.type(torch.int64), num_classes=int(deg.max() + 1))
        if data.x is None:
            data.x = deg
        else:
            data.x = torch.concat((data.x, deg.unsqueeze(1)), -1)
        return data


class AddCoord(BaseTransform):
    def __call__(self, data):
        data.pos = data.x[:, -2:]
        return data


class AddNormCoord(BaseTransform):
    def __call__(self, data):
        data.pos = data.x[:, -2:]
        return data


class NormalizeCoord(BaseTransform):
    def __call__(self, data):
        data.x[:, -2] = data.x[:, -2] / data.x[:, -2].max()
        data.x[:, -1] = data.x[:, -1] / data.x[:, -1].max()
        return data


class LabelRename(BaseTransform):
    def __call__(self, data):
        data.y = data.edge_label.type(torch.float32)
        return data