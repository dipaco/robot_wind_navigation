import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric import nn as gnn


def create_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False, conv_type='node', output_layer=True):
    if conv_type == 'node':
        return _create_node_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    if conv_type == 'edge':
        return _create_edge_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    else:
        raise ValueError(f'Wrong Graph Convolution type: {conv_type}. Try [node, edge].')


def _create_node_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=True, output_layer=True):
    all_layers = []
    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (gnn.GraphConv(in_channels=in_dim, out_channels=hidden_dim, bias=False), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=out_dim, bias=False),
                 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=hidden_dim, bias=False), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))

    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def _create_edge_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False, output_layer=True):
    all_layers = []

    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*in_dim, out_features=hidden_dim, bias=False), aggr='add'), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*hidden_dim, out_features=out_dim, bias=False), aggr='add'),
                 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*hidden_dim, out_features=hidden_dim, bias=False), aggr='add'), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=True))

    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def create_output_mlp(in_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(in_dim, in_dim // 4), nn.BatchNorm1d(in_dim // 4), nn.ReLU(inplace=True),
        nn.Linear(in_dim // 4, in_dim // 16), nn.BatchNorm1d(in_dim // 16), nn.ReLU(inplace=True),
        nn.Linear(in_dim // 16, out_dim)
    )

    return mlp


def create_mlp(input_dim, output_dim, hidden_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))

    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
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
    elif formation_type in range(2, 6):  # grid 2x2
        G = nx.grid_graph(dim=[formation_type, formation_type])
        formation_ref = 0.5*np.array(G.nodes).astype(float)
        G = nx.relabel.relabel_nodes(G, mapping=dict(zip(G.nodes, range(formation_type**2))))

    elif formation_type == 6:    # platoon
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


def normalize(v, axis=-1):
    eps = 1e-8
    norm = np.linalg.norm(v, axis=axis, keepdims=True)

    norm_v = v / (norm + eps)

    idx = np.where(norm == 0.0)[0]
    norm_v[idx, :] = 0.0

    return norm_v


def zero_weight_init(m):

    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.0)
    elif isinstance(m, gnn.GraphConv):
        m.lin_l.weight.data.fill_(0.0)
    elif isinstance(m, gnn.EdgeConv):
        m.nn.weight.data.fill_(0.0)

