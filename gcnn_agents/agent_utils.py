import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric import nn as gnn
from torch_geometric.nn import knn_graph as torch_knn_graph

from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul


def create_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False, conv_type='node', output_layer=True):
    if conv_type == 'node':
        return _create_node_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    elif conv_type == 'edge':
        return _create_edge_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    elif conv_type == 'attention':
        return _create_attention_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    elif conv_type == 'gcn':
        return _create_gcn_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    elif conv_type == 'tag':
        return _create_tag_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    elif conv_type == 'dna':
        return _create_dna_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    elif conv_type == 'rel_pos':
        return _create_rel_pose_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    elif conv_type == 'rel_pos_hack':
        return _create_rel_pose_hack_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity, norm, output_layer)
    else:
        raise ValueError(f'Wrong Graph Convolution type: {conv_type}. Try [node, edge].')


def _create_gcn_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=True, output_layer=True):
    all_layers = []
    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (gnn.GCNConv(in_channels=in_dim, out_channels=hidden_dim), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (gnn.GCNConv(in_channels=hidden_dim, out_channels=out_dim, bias=True), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (gnn.GCNConv(in_channels=hidden_dim, out_channels=hidden_dim), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))

    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def _create_tag_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=True, output_layer=True):
    all_layers = []
    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (gnn.TAGConv(in_channels=in_dim, out_channels=hidden_dim, K=2), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (gnn.TAGConv(in_channels=hidden_dim, out_channels=out_dim, K=2, bias=True), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (gnn.TAGConv(in_channels=hidden_dim, out_channels=hidden_dim, K=2), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))

    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def _create_attention_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=True, output_layer=True):
    all_layers = []
    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (gnn.GATConv(in_channels=in_dim, out_channels=hidden_dim // 4, heads=4), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (gnn.GATConv(in_channels=hidden_dim, out_channels=out_dim, heads=1, bias=True), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (gnn.GATConv(in_channels=hidden_dim, out_channels=hidden_dim // 4, heads=4), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))

    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def _create_rel_pose_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=True, output_layer=True):
    all_layers = []
    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (GraphRelPosConv(in_channels=in_dim, out_channels=hidden_dim, dim=2, aggr='mean'), 'x, edge_index, rel_pos -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (GraphRelPosConv(in_channels=hidden_dim, out_channels=out_dim, dim=2, aggr='mean'), 'x, edge_index, rel_pos -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (GraphRelPosConv(in_channels=hidden_dim, out_channels=hidden_dim, dim=2, aggr='mean'), 'x, edge_index, rel_pos -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))

    trunk = gnn.Sequential('x, edge_index, rel_pos', all_layers)

    return trunk


def _create_rel_pose_hack_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=True, output_layer=True):
    all_layers = []
    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1

    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (GraphRelPosHackConv(in_channels=in_dim, out_channels=hidden_dim, dim=2, aggr='max'), 'x, edge_index, rel_pos -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (GraphRelPosHackConv(in_channels=hidden_dim, out_channels=out_dim, dim=2, aggr='max'), 'x, edge_index, rel_pos -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (GraphRelPosHackConv(in_channels=hidden_dim, out_channels=hidden_dim, dim=2, aggr='max'), 'x, edge_index, rel_pos -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))

    trunk = gnn.Sequential('x, edge_index, rel_pos', all_layers)

    return trunk


def _create_node_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=True, output_layer=True):
    all_layers = []
    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (gnn.GraphConv(in_channels=in_dim, out_channels=hidden_dim, aggr='mean'), 'x, edge_index, edge_weights -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=out_dim, aggr='mean'),
                 'x, edge_index, edge_weights -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (gnn.GraphConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr='mean'), 'x, edge_index, edge_weights -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))

    trunk = gnn.Sequential('x, edge_index, edge_weights', all_layers)

    return trunk


def _create_edge_gcnn(in_dim, out_dim, hidden_dim, hidden_depth, non_linearity=nn.ReLU, norm=False, output_layer=True):
    all_layers = []

    num_layers = hidden_depth + 2 if output_layer else hidden_depth + 1
    for i in range(num_layers):
        if i == 0:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*in_dim, out_features=hidden_dim), aggr='mean'), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))
        elif output_layer and i == num_layers - 1:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*hidden_dim, out_features=out_dim, bias=True), aggr='mean'),
                 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
        else:
            all_layers.append(
                (gnn.EdgeConv(nn.Linear(in_features=2*hidden_dim, out_features=hidden_dim), aggr='mean'), 'x, edge_index -> x')
            )
            if norm: all_layers.append(gnn.BatchNorm(in_channels=hidden_dim, track_running_stats=True))
            all_layers.append(non_linearity(inplace=False))

    trunk = gnn.Sequential('x, edge_index', all_layers)

    return trunk


def create_output_mlp(in_dim, out_dim, last_layer_bias=False):
    mlp = nn.Sequential(
        nn.Linear(in_dim, in_dim // 4), nn.ReLU(inplace=False),
        nn.Linear(in_dim // 4, out_dim, bias=last_layer_bias)
    )

    return mlp


def create_input_mlp(in_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(in_dim, out_dim // 4), nn.ReLU(inplace=False),
        nn.Linear(out_dim // 4, out_dim // 2), nn.ReLU(inplace=False),
        nn.Linear(out_dim // 2, out_dim, bias=False)
    )

    return mlp


def create_mlp(input_dim, output_dim, hidden_dim, hidden_depth, output_layer=True):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=False)]
        for i in range(hidden_depth):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=False)]

        if output_layer:
            mods.append(nn.Linear(hidden_dim, output_dim, bias=False))

    trunk = nn.Sequential(*mods)
    return trunk


def get_formation_conf(formation_type, robot_distance=0.25):
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
    elif formation_type in range(2, 9):  # grid 2x2
        G = nx.grid_graph(dim=[formation_type, formation_type])
        formation_ref = np.array(G.nodes).astype(float) * robot_distance #/ 2.0 # / (formation_type - 1)
        G = nx.relabel.relabel_nodes(G, mapping=dict(zip(G.nodes, range(formation_type**2))))

    elif formation_type == 9:    # platoon
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


def min_distance_graph(robot_loc, bs, num_nodes, th=1.0):

    device = robot_loc.device
    aux = torch.arange(bs, device=device)[:, None].repeat(1, num_nodes).reshape(-1)
    batch_filter = (aux[None, :] == aux[:, None])
    dist_mat = torch.cdist(robot_loc, robot_loc)
    aux = dist_mat * batch_filter
    edges = torch.stack(torch.where(torch.logical_and(aux > 0.0, aux <= th)))

    return edges


def delaunay_graph(robot_loc, bs, num_nodes):
    batch_idx = torch.arange(bs, device=robot_loc.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    raise NotImplementedError()


def knn_graph(robot_loc, k, bs, num_nodes):
    batch_idx = torch.arange(bs, device=robot_loc.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    edges = torch_knn_graph(robot_loc, k, batch_idx)

    return edges


class GraphRelPosConv(gnn.MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 dim: int,
                 aggr: str = 'add',
                 bias: bool = True,
                 **kwargs):
        super(GraphRelPosConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0] + dim, out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, rel_pos: Union[Tensor, OptPairTensor],
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=rel_pos, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else torch.cat([x_j, edge_weight], dim=-1)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphRelPosHackConv(gnn.MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 dim: int,
                 aggr: str = 'add',
                 bias: bool = True,
                 **kwargs):
        super(GraphRelPosHackConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0] + dim, out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, rel_pos: Union[Tensor, OptPairTensor],
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=rel_pos, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else torch.cat([x_j, edge_weight], dim=-1)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)