import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn


def create_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False):
    all_layers = []
    for i in range(hidden_depth + 2):
        if i == 0:
            all_layers.append(
                (gnn.GraphConv(in_channels=in_dim, out_channels=hidden_dim, node_dim=1), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))
        elif i == hidden_depth + 1:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=out_dim, node_dim=1),
                 'x, edge_index -> x')
            )
        else:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=hidden_dim, node_dim=1), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))


    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk
