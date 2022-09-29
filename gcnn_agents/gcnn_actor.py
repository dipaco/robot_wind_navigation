import numpy as np
import torch
import os
import sys
import math
from torch import nn
import torch.nn.functional as F
import networkx as nx
import torch_scatter
from torch import distributions as pyd
from torch_geometric import nn as gnn
from torch_geometric.utils import to_undirected, add_self_loops
from .agent_utils import create_gcnn, create_mlp, create_output_mlp, zero_weight_init, min_distance_graph, delaunay_graph, knn_graph, create_input_mlp
from torch_cluster import knn

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
                 log_std_bounds, use_ns_regularization, conv_type, input_batch_norm, formation_type, ignore_neighbors, ignore_neighbors_at_testing, graph_type, num_delays, use_time_delays, num_neighbors):
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
        self.use_output_mlp = True
        self.use_input_mlp = False
        self.residual = False
        self.ignore_neighbors = ignore_neighbors
        self.ignore_neighbors_at_testing = ignore_neighbors_at_testing
        self.graph_type = graph_type
        self.use_time_delays = use_time_delays
        self.num_delays = num_delays if self.use_time_delays else 1
        self.num_neighbors = num_neighbors

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
            self.input_norm_layer = nn.BatchNorm1d(gnn_obs_dim, track_running_stats=True)

        if self.use_ns_regularization:
            self.mlp_mag = create_output_mlp(self.hidden_dim, 1, last_layer_bias=True)

            '''num_excluded_variables = 3   # p_x_robot, p_y_robot, v_x_robot, v_y_robot
            self.num_ns_input = gnn_obs_dim - num_excluded_variables
            self.num_ns_output = (self.gnn_action_dim * (self.gnn_action_dim + 1)) // 2
            self.ns_branch = create_gcnn(self.num_ns_input, self.num_ns_output, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type)
            # we do not pass the global position to the network
            trunk_input_size = gnn_obs_dim + self.num_ns_output - self.gnn_action_dim * self.num_delays'''

        # we do not pass the global position to the network
        trunk_input_size = gnn_obs_dim - self.gnn_action_dim * self.num_delays

        if self.use_input_mlp:
            self.encoder = create_input_mlp(trunk_input_size, self.hidden_dim)

        if self.ignore_neighbors:
            if not self.use_input_mlp:
                self.trunk = create_mlp(trunk_input_size, 2 * self.gnn_action_dim, self.hidden_dim, self.hidden_depth, output_layer=not self.use_output_mlp)
            else:
                self.trunk = create_mlp(self.hidden_dim, 2 * self.gnn_action_dim, self.hidden_dim, self.hidden_depth - 1, output_layer=not self.use_output_mlp)
        else:
            if not self.use_input_mlp:
                self.trunk = create_gcnn(trunk_input_size, 2 * self.gnn_action_dim, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=not self.use_output_mlp)
            else:
                self.trunk = create_gcnn(self.hidden_dim, 2 * self.gnn_action_dim, self.hidden_dim, self.hidden_depth - 1, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=not self.use_output_mlp)

        if self.use_output_mlp:
            self.mlp_trunk = create_output_mlp(self.hidden_dim, 2 * self.gnn_action_dim)

        self.outputs = dict()
        #self.apply(zero_weight_init)
        self.cc = 1

    def forward(self, obs):

        device = obs.device
        eps = 1e-10

        bs = obs.shape[0]
        input_features = obs.view(bs * self.num_nodes, self.obs_dim // self.num_nodes)

        if self.input_batch_norm:
            # Normalize the input features
            input_features = self.input_norm_layer(input_features)

        #TODO: Also extact the locations for other time steps
        robot_loc = input_features[:, :self.gnn_action_dim]

        if self.use_time_delays:
            # Removes all the locations from the input features
            aux_s = input_features.shape[0]
            input_features = input_features.reshape(aux_s, self.num_delays, -1)[:, :, self.gnn_action_dim:]
            input_features = input_features.reshape(aux_s, -1)
        else:
            input_features = input_features[:, self.gnn_action_dim:]

        #batch_idx = torch.arange(bs, device=input_features.device).view(-1, 1).repeat(1, self.num_nodes).view(-1)

        # FIXME: This can be definitely done better using the Batch Class from torch_geometric
        if self.ignore_neighbors or (not self.training and self.ignore_neighbors_at_testing):
            edges = torch.stack(2 * [torch.arange(self.num_nodes, device=device)]).long()
            edges = torch.cat([edges + i * self.num_nodes for i in range(bs)], dim=-1)
        else:
            if self.graph_type == 'formation':
                G = self.G
                edges = to_undirected(torch.tensor([e for e in G.edges], device=device).long().T)
                edges = torch.cat([edges + i * self.num_nodes for i in range(bs)], dim=-1)
            elif self.graph_type == 'complete':
                G = nx.complete_graph(self.num_nodes)
                edges = to_undirected(torch.tensor([e for e in G.edges], device=device).long().T)
                edges = torch.cat([edges + i * self.num_nodes for i in range(bs)], dim=-1)
            elif self.graph_type == 'knn':
                edges = knn_graph(robot_loc, self.num_neighbors, bs, self.num_nodes)
            elif self.graph_type == 'delaunay':
                edges = delaunay_graph(robot_loc, bs, self.num_nodes)
            elif self.graph_type == 'min_dist':
                r_th = 1.0
                edges = min_distance_graph(robot_loc, bs, self.num_nodes, r_th)
            else:
                raise ValueError(f'Wrong graph type: {self.graph_type}. Provide a value in [formation, complete, knn]')

            #if self.conv_type == 'edge':
            #    edges, _ = add_self_loops(edges, num_nodes=bs*self.num_nodes)

        if self.use_input_mlp:
            input_features = self.encoder(input_features)

        if self.ignore_neighbors:
            # we do not pass the global position to the network
            net_args = (input_features, )
        else:
            # we do not pass the global position to the network
            # we need to pass the edges when the networks is a GNN
            if self.conv_type == 'node':
                edge_weights = torch.exp(-(robot_loc[edges[1]] - robot_loc[edges[0]]).norm(dim=-1))
                edge_weights = torch.ones_like(edge_weights)
                net_args = (input_features, edges, edge_weights)
            elif self.conv_type == 'rel_pos':
                rel_pos = robot_loc[edges[1]] - robot_loc[edges[0]]
                net_args = (input_features, edges, rel_pos)
            elif self.conv_type == 'rel_pos_hack':
                rel_pos = robot_loc[edges[1]] - robot_loc[edges[0]]
                net_args = (input_features, edges, rel_pos)
            else:
                net_args = (input_features, edges)

        if self.residual:
            out_latent = self.trunk(*net_args) + input_features
        else:
            out_latent = self.trunk(*net_args)

        if self.use_output_mlp:
            out = self.mlp_trunk(out_latent)

        mu, log_std = out.chunk(2, dim=-1)

        if self.use_ns_regularization:

            ns_loss = self._get_ns_loss(mu, robot_loc, input_features, edges)

            # TODO: What we need is w_bar, we need to fix this
            # FIXME: This has changed a lot when I started removing the position form the input features
            '''p_idx = 6
            w_idx = 4
            x_idx = 0
            x = input_features[:, x_idx:x_idx + self.gnn_action_dim]  # Get the robots positions
            w = input_features[:, w_idx:w_idx + self.gnn_action_dim]  # Get average wind vector
            P = input_features[:, p_idx:p_idx + 1]  # Get the pressure

            turb_energy = self.ns_branch(input_features[:, self.action_dim:self.num_ns_input], edges)
            input_features = torch.cat([input_features, turb_energy], dim=-1)

            ns_loss = self._get_ns_loss(w, x, P, turb_energy, edges)'''

            #
            mu = F.softplus(self.mlp_mag(out_latent)) * mu
        else:
            ns_loss = None

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

        nets = self.trunk if self.ignore_neighbors else self.trunk.nns
        for i, m in enumerate(nets):
            if type(m) in [gnn.GCNConv, nn.Linear]:
                logger.log_param(f'train_actor/fc{i}', m, step)

    def _spatial_derivative(self, y, x, edges):

        # FIXME: I am going to hardcode the faces here. We need to look at an strategy for triangulation.
        device = x.device
        d = x.shape[-1]
        x = x[:, 0]
        y = y[:, 0]

        eps = 1e-8

        no_loop_edges = edges[:, edges[0] != edges[1]]

        edge_coords = x[no_loop_edges.T]
        edge_f_values = y[no_loop_edges.T].permute(0, 2, 1)
        #face_f_values = torch.ones_like(face_f_values).permute(0, 2, 1)

        edge_length = (edge_coords[:, 1] - edge_coords[:, 0]).norm(dim=-1)

        A_T = torch.cat([edge_coords, torch.ones(edge_coords.shape[:2] + (1, ), device=device)], dim=-1)
        A = A_T.permute(0, 2, 1)

        # Thankfully we know a formula to invert 2x2 matrices
        A_T_A = A_T @ A
        det = 1/(A_T_A[:, 0, 0]*A_T_A[:, 1, 1] - A_T_A[:, 0, 1]*A_T_A[:, 1, 0])
        adj = torch.zeros_like(A_T_A)
        adj[:, 0, 0] = A_T_A[:, 1, 1]
        adj[:, 1, 1] = A_T_A[:, 0, 0]
        adj[:, 0, 1] = -A_T_A[:, 1, 0]
        adj[:, 1, 0] = -A_T_A[:, 0, 1]
        A_T_A_inv = det[:, None, None] * adj @ A_T
        B = torch.eye(d + 1, device=device)[None].repeat(A.shape[0], 1, 1)[:, :, :d]
        H = edge_f_values @ A_T_A_inv @ B
        #torch.linalg.lstsq(A, B).solution == A.pinv() @ B

        #H = edge_f_values @ torch.linalg.lstsq(A, B).solution

        derivative = torch_scatter.scatter_mean(src=H, index=no_loop_edges[0], dim=0)

        #D = torch.linalg.pinv(A_T @ A) @ A_T
        #H = (face_f_values.view(-1, 3) @ D) @ torch.eye(d + 1, device=device)[:, :d]

        return derivative

    def _spatial_derivative_3(self, y, x, edges):

        x = x[:, 0]
        y = y[:, 0]

        eps = 1e-8

        no_loop_edges = edges[:, edges[0] != edges[1]]

        import pdb
        pdb.set_trace()
        x_center = x[no_loop_edges[0]]
        x_neighbors = x[no_loop_edges[1]]
        d =  x_center

        # Approximate the derivative using the neighbors on each node of the grad
        delta_y = (y[no_loop_edges[1]] - y[no_loop_edges[0]]).permute(0, 2, 1)
        delta_x = (x[no_loop_edges[1]] - x[no_loop_edges[0]])
        derivative = delta_y / (delta_x + eps)

        H = torch_scatter.scatter_mean(src=derivative, index=no_loop_edges[0], dim=0)
        return H

    def _spatial_derivative_2(self, y, x, edges):

        eps = 1e-8

        no_loop_edges = edges[:, edges[0] != edges[1]]

        # Approximate the derivative using the neighbors on each node of the grad
        delta_y = (y[no_loop_edges[1]] - y[no_loop_edges[0]]).permute(0, 2, 1)
        delta_x = (x[no_loop_edges[1]] - x[no_loop_edges[0]])
        derivative = delta_y / (delta_x + eps)

        H = torch_scatter.scatter_mean(src=derivative, index=no_loop_edges[0], dim=0)
        return H

    def _get_symetric_matrix_from_unique_values(self, vals):

        d = (-1 + math.sqrt(1 + 4*2*vals.shape[-1])) / 2
        assert d.is_integer(), f"We can't create a symmetric matrix out of {vals.shape[-1]} unique values."
        d = int(d)

        bs, n = vals.shape[:2]
        M = torch.zeros(bs, d, d, device=vals.device)
        i, j = torch.triu_indices(d, d)
        M[:, i, j] = vals
        M = M.permute(0, 2, 1)
        M[:, i, j] = vals

        return M

    def _get_ns_loss(self, v, pos, input_features, edges):

        #FIXME: Take care of removing the self_loops if they exist

        device = v.device
        d = self.gnn_action_dim

        p = input_features[:, 0:1]

        v_ext = v[..., None, :].repeat(1, d, 1)
        pos_ext = pos[..., None, :].repeat(1, d, 1)
        p_ext = p[..., None, :].repeat(1, d, 1)

        nabla_v = self._spatial_derivative(v_ext, pos_ext, edges)
        nabla_p = self._spatial_derivative(p_ext, pos_ext, edges)

        # Navier stokes equation
        ns = (nabla_v @ v[..., None]) + nabla_p.permute(0, 2, 1)
        ns_loss = (ns.squeeze(-1) ** 2).sum(dim=-1).mean()

        return ns_loss


        rho = 1.184
        u_prime_bar_matrix = self._get_symetric_matrix_from_unique_values(u_prime_bar)
        P_matrix = torch.eye(d, device=device)[None, ...] * P[..., None]
        f = torch.zeros_like(u_bar)
        u_bar_matrix = u_bar[..., None, :].repeat(1, d, 1)
        pos_matrix = pos[..., None, :].repeat(1, d, 1)

        # TODO: Fix derivative computation
        # https://github.com/ZichaoLong/aTEAM/blob/master/nn/modules/Interpolation.py
        #import pdb
        #pdb.set_trace()

        dxj_u_bar_matrix = self._spatial_derivative(u_bar_matrix, pos_matrix, edges)
        dxj_forces_matrix = self._spatial_derivative(-(P_matrix + rho * u_prime_bar_matrix), pos_matrix, edges)

        RANS = (rho * u_bar_matrix * dxj_u_bar_matrix - dxj_forces_matrix).sum(dim=-1)
        #rans_loss = (RANS ** 2).sum(dim=-1).mean()
        rans_loss = (RANS ** 2).sum(dim=-1).sqrt().mean()

        return rans_loss

    def _get_ns_loss_old(self, u_bar, pos, P, u_prime_bar, edges):

        #FIXME: Take care of removing the self_loops if they exist

        device = u_prime_bar.device
        d = self.gnn_action_dim
        rho = 1.184
        u_prime_bar_matrix = self._get_symetric_matrix_from_unique_values(u_prime_bar)
        P_matrix = torch.eye(d, device=device)[None, ...] * P[..., None]
        f = torch.zeros_like(u_bar)
        u_bar_matrix = u_bar[..., None, :].repeat(1, d, 1)
        pos_matrix = pos[..., None, :].repeat(1, d, 1)

        # TODO: Fix derivative computation
        # https://github.com/ZichaoLong/aTEAM/blob/master/nn/modules/Interpolation.py
        #import pdb
        #pdb.set_trace()

        dxj_u_bar_matrix = self._spatial_derivative(u_bar_matrix, pos_matrix, edges)
        dxj_forces_matrix = self._spatial_derivative(-(P_matrix + rho * u_prime_bar_matrix), pos_matrix, edges)

        RANS = (rho * u_bar_matrix * dxj_u_bar_matrix - dxj_forces_matrix).sum(dim=-1)
        #rans_loss = (RANS ** 2).sum(dim=-1).mean()
        rans_loss = (RANS ** 2).sum(dim=-1).sqrt().mean()

        return rans_loss
