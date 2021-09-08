import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
import networkx as nx
from torch import distributions as pyd
from torch_geometric import nn as gnn
from torch_geometric.utils import to_undirected
from .agent_utils import create_gcnn

import utils


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class GCNNDiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, num_nodes, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.num_nodes = num_nodes
        self.log_std_bounds = log_std_bounds
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        assert self.obs_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                              f' do not divide the observation space size ({self.obs_dim}.)'
        assert self.action_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                              f' do not divide the action space size ({self.action_dim}.)'

        # specify the formation graph
        # TODO: this must come from configuration parameters
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_nodes))
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        # create all the convolutional layers
        gnn_obs_dim = self.obs_dim // self.num_nodes
        gnn_action_dim = self.action_dim // self.num_nodes
        self.trunk = create_gcnn(gnn_obs_dim, 2*gnn_action_dim, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU)

        self.outputs = dict()
        #self.apply(utils.weight_init)

    def forward(self, obs):

        bs = obs.shape[0]
        input_features = obs.view(bs, self.num_nodes, self.obs_dim // self.num_nodes)

        # FIXME: This can be definitely done better using the Batch Class from torch_geometric
        edges = to_undirected(torch.tensor([e for e in self.G.edges], device=input_features.device).long().T)

        mu, log_std = self.trunk(input_features, edges).chunk(2, dim=-1)

        # We can flatten the distribution along the nodes of the graph because the Covariance matrix is diagonal
        mu = mu.contiguous().view(bs, -1)
        log_std = log_std.contiguous().view(bs, -1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk.nns):
            if type(m) in [gnn.GCNConv]:
                logger.log_param(f'train_actor/fc{i}', m, step)