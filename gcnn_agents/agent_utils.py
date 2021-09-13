import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric import nn as gnn


def create_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False, conv_type='node'):
    if conv_type == 'node':
        return _create_node_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm)
    if conv_type == 'edge':
        return _create_edge_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm)
    else:
        raise ValueError(f'Wrong Graph Convolution type: {conv_type}. Try [node, edge].')


def _create_node_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False):
    all_layers = []
    for i in range(hidden_depth + 2):
        if i == 0:
            all_layers.append(
                (gnn.GraphConv(in_channels=in_dim, out_channels=hidden_dim, bias=False, node_dim=1), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))
        elif i == hidden_depth + 1:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=out_dim, bias=False, node_dim=1),
                 'x, edge_index -> x')
            )
        else:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=hidden_dim, bias=False, node_dim=1), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))


    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def _create_edge_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False):
    all_layers = []
    for i in range(hidden_depth + 2):
        if i == 0:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*in_dim, out_features=hidden_dim, bias=False), aggr='add', node_dim=1), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))
        elif i == hidden_depth + 1:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*hidden_dim, out_features=out_dim, bias=False), aggr='add', node_dim=1),
                 'x, edge_index -> x')
            )
        else:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*hidden_dim, out_features=hidden_dim, bias=False), aggr='add', node_dim=1), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))


    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def get_formation_conf(formation_type):
    G = nx.Graph()
    if formation_type == 0:  # small triangle
        formation_ref = np.array([[0.0, 0.0], [1.0, 0.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])
        # specify the formation graph
        G.add_nodes_from(range(formation_ref.shape[0]))
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    elif formation_type == 1:    # triangle
        formation_ref = np.array(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4], [0.0, 5.0], [1, 0], [1, 1], [1, 2], [1, 3],
             [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1], [5, 0]]) * 1.5
        G.add_nodes_from(range(formation_ref.shape[0]))
        G.add_edges_from(
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (6, 7), (7, 8),
             (8, 9), (9, 10), (11, 6), (12, 7), (13, 8), (14, 9), (11, 12), (12, 13), (13, 14), (15, 11), (16, 12),
             (17, 13), (15, 16), (16, 17), (18, 15), (19, 16), (18, 19), (20, 18)])
    elif formation_type == 2:    # platoon
        formation_ref = np.array(
            [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2, 2], [2.5, 2.5], [3, 3], [3.5, 3.5], [4, 4],
             [4.5, 4.5], [5, 5], [5.5, 5.5], [6, 6], [6.5, 6.5], [7, 7], [7.5, 7.5], [8, 8], [8.5, 8.5], [9, 9],
             [9.5, 9.5], [10, 10]])
        G.add_nodes_from(range(formation_ref.shape[0]))
        G.add_edges_from(
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)])
    elif formation_type == 3:  # grid 3x3
        formation_ref = 2/3*np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])
        G.add_nodes_from(range(formation_ref.shape[0]))
        G.add_edges_from(
            [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (3, 6), (1, 4), (4, 7),
             (2, 5), (5, 8)])
    elif formation_type == 4:  # grid 4x4
        formation_ref = 0.5*np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [0.0, 2.0], [1.0, 2.0],
             [2.0, 2.0], [3.0, 2.0], [0.0, 3.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0]])
        G.add_nodes_from(range(formation_ref.shape[0]))
        G.add_edges_from(
            [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (12, 13),
             (13, 14), (14, 15), (0, 4), (4, 8), (8, 12), (1, 5), (5, 9), (9, 13), (2, 6), (6, 10), (10, 14),
             (3, 7), (7, 11), (11, 15)])
    else:
        raise ValueError(f'Invalid formation type {self.config.formation_params.formation_type}. Tyr [0-> "small triangle, "1 -> "Triangle", 2 -> "Platoon"].')

    return formation_ref, G