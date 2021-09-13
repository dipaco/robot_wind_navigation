import numpy as np
import torch
import os
import sys
import math
from torch import nn
import torch.nn.functional as F
import networkx as nx
from torch import distributions as pyd
from torch_geometric import nn as gnn
from torch_geometric.utils import to_undirected
from .agent_utils import create_gcnn

import utils

__BASE_FOLDER__ = os.path.dirname(os.path.abspath(__file__))
dist_package_folder = os.path.join(__BASE_FOLDER__, '../gcnn_agents/')
sys.path.append(dist_package_folder)
from agent_utils import get_formation_conf


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
                 log_std_bounds, use_ns_regularization, conv_type, input_batch_norm, formation_type):
        super().__init__()

        self.num_nodes = num_nodes
        self.log_std_bounds = log_std_bounds
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_ns_regularization = use_ns_regularization
        self.conv_type = conv_type
        self.input_batch_norm = input_batch_norm
        self.formation_type = formation_type

        assert self.obs_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                              f' do not divide the observation space size ({self.obs_dim}.)'
        assert self.action_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                              f' do not divide the action space size ({self.action_dim}.)'

        self.G = None
        _, self.G = get_formation_conf(self.formation_type)

        # create all the convolutional layers
        gnn_obs_dim = self.obs_dim // self.num_nodes
        self.gnn_action_dim = self.action_dim // self.num_nodes

        if self.input_batch_norm:
            self.input_norm_layer = nn.BatchNorm1d(self.obs_dim, track_running_stats=True)

        if self.use_ns_regularization:
            num_dynamic_variables = 4   # p_x_robot, p_y_robot, v_x_robot, v_y_robot
            self.num_ns_input = gnn_obs_dim - num_dynamic_variables
            self.num_ns_output = (self.gnn_action_dim * (self.gnn_action_dim + 1)) // 2

            self.ns_branch = create_gcnn(self.num_ns_input, self.num_ns_output, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type)
            self.trunk = create_gcnn(gnn_obs_dim + self.num_ns_output, 2*self.gnn_action_dim, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type)
        else:
            self.trunk = create_gcnn(gnn_obs_dim, 2*self.gnn_action_dim, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type)

        self.outputs = dict()
        #self.apply(utils.weight_init)

    def forward(self, obs):

        if self.input_batch_norm:
            # Normalize the input features
            obs = self.input_norm_layer(obs)

        bs = obs.shape[0]
        input_features = obs.view(bs, self.num_nodes, self.obs_dim // self.num_nodes)

        # FIXME: This can be definitely done better using the Batch Class from torch_geometric
        edges = to_undirected(torch.tensor([e for e in self.G.edges], device=input_features.device).long().T)

        if self.use_ns_regularization:

            # TODO: What we need is w_bar, we need to fix this
            w = input_features[:, :, -self.num_ns_input:-self.num_ns_input+self.gnn_action_dim]
            P = input_features[:, :, -1:]

            turb_energy = self.ns_branch(input_features[:, :, -self.num_ns_input:], edges)
            input_features = torch.cat([input_features, turb_energy], dim=-1)

            ns_loss = self._get_ns_loss(w, P, turb_energy, edges)
        else:
            ns_loss = None

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
        return dist, ns_loss

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk.nns):
            if type(m) in [gnn.GCNConv]:
                logger.log_param(f'train_actor/fc{i}', m, step)

    def _spatial_derivative(self, y, edges):
        return y

    def _get_symetric_matrix_from_unique_values(self, vals):

        d = (-1 + math.sqrt(1 + 4*2*vals.shape[-1])) / 2
        assert d.is_integer(), f"We can't create a symmetric matrix out of {vals.shape[-1]} unique values."
        d = int(d)

        bs, n = vals.shape[:2]
        M = torch.zeros(bs, n, d, d, device=vals.device)
        i, j = torch.triu_indices(d, d)
        M[:, :, i, j] = vals
        M = M.permute(0, 1, 3, 2)
        M[:, :, i, j] = vals

        return M

    def _get_ns_loss(self, u_bar, P, u_prime_bar, edges):

        device = u_prime_bar.device
        d = u_bar.shape[-1]
        rho = 1.184
        u_prime_bar_matrix = self._get_symetric_matrix_from_unique_values(u_prime_bar)
        P_matrix = torch.eye(d, device=device)[None, None, ...] * P[..., None]
        u_bar_matrix = u_bar[..., None, :].repeat(1, 1, d, 1)

        # TODO: Fix derivative computation
        #import pdb
        #pdb.set_trace()

        dxj_u_bar_matrix = self._spatial_derivative(u_bar_matrix, edges)
        dxj_forces_matrix = self._spatial_derivative(-(P_matrix + rho * u_prime_bar_matrix), edges)

        rans_loss = (rho * u_bar_matrix * dxj_u_bar_matrix - dxj_forces_matrix).abs().mean(dim=[0, 1]).sum()

        #import pdb
        #pdb.set_trace()

        return rans_loss
