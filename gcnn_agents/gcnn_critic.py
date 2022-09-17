import numpy as np
import torch
import os
import sys
import networkx as nx
from torch import nn
from .agent_utils import create_gcnn, create_mlp, create_output_mlp, zero_weight_init, min_distance_graph, delaunay_graph, knn_graph, create_input_mlp
from torch_geometric import nn as gnn
from torch_geometric.utils import to_undirected, add_self_loops
from torch_cluster import knn
import torch.nn.functional as F

import utils

__BASE_FOLDER__ = os.path.dirname(os.path.abspath(__file__))
dist_package_folder = os.path.join(__BASE_FOLDER__, '../gcnn_agents/')
sys.path.append(dist_package_folder)
from agent_utils import get_formation_conf

class GCNNDoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, num_nodes, obs_dim, action_dim, hidden_dim, hidden_depth, conv_type, input_batch_norm, formation_type, ignore_neighbors, ignore_neighbors_at_testing, graph_type, num_delays, use_time_delays):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.obs_dim = obs_dim
        self.action_dim = action_dim
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

        assert self.obs_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                                   f' do not divide the observation space size ({self.obs_dim}.)'
        assert self.action_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                                      f' do not divide the action space size ({self.action_dim}.)'

        # create all the convolutional layers
        gnn_obs_dim = self.obs_dim // self.num_nodes
        self.gnn_action_dim = self.action_dim // self.num_nodes

        if self.input_batch_norm:
            self.input_norm_layer_obs = nn.BatchNorm1d(gnn_obs_dim, track_running_stats=True)
            self.input_norm_layer_action = nn.BatchNorm1d(self.action_dim // self.num_nodes, track_running_stats=True)

        #NOTE: we do not pass the global position to the network
        # we add gnn_action_dim to indicate that we are removing dimensions from the observation space
        # but adding the action dimension to the Q network's input
        trunk_input_size = (gnn_obs_dim - self.gnn_action_dim * self.num_delays) + self.gnn_action_dim

        self.encoder1 = create_input_mlp(trunk_input_size, self.hidden_dim)
        self.encoder2 = create_input_mlp(trunk_input_size, self.hidden_dim)

        if self.ignore_neighbors:
            if not self.use_input_mlp:
                self.Q1 = create_mlp(trunk_input_size, 1, self.hidden_dim, self.hidden_depth, output_layer=not self.use_output_mlp)
                self.Q2 = create_mlp(trunk_input_size, 1, self.hidden_dim, self.hidden_depth, output_layer=not self.use_output_mlp)
            else:
                self.Q1 = create_mlp(self.hidden_dim, 1, self.hidden_dim, self.hidden_depth - 1, output_layer=not self.use_output_mlp)
                self.Q2 = create_mlp(self.hidden_dim, 1, self.hidden_dim, self.hidden_depth - 1, output_layer=not self.use_output_mlp)
        else:
            if not self.use_input_mlp:
                self.Q1 = create_gcnn(trunk_input_size, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=not self.use_output_mlp)
                self.Q2 = create_gcnn(trunk_input_size, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=not self.use_output_mlp)
            else:
                self.Q1 = create_gcnn(self.hidden_dim, 1, self.hidden_dim, self.hidden_depth - 1, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=not self.use_output_mlp)
                self.Q2 = create_gcnn(self.hidden_dim, 1, self.hidden_dim, self.hidden_depth - 1, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=not self.use_output_mlp)

        if self.use_output_mlp:
            # FIXME: (dipaco) I am unsure if this actually works. I think the dimensions for the output mlp are not right
            self.mlp_Q1 = create_output_mlp(self.hidden_dim, 1)
            self.mlp_Q2 = create_output_mlp(self.hidden_dim, 1)

        self.G = None
        _, self.G = get_formation_conf(self.formation_type)

        self.outputs = dict()
        #self.apply(zero_weight_init)

    def forward(self, obs, action):

        device = obs.device

        bs = obs.shape[0]

        assert obs.size(0) == action.size(0)

        if self.input_batch_norm:
            # Normalize the input features
            obs = self.input_norm_layer_obs(obs.view(bs * self.num_nodes, -1))

            # Normalize the input features
            action = self.input_norm_layer_action(action.view(bs * self.num_nodes, -1))

        obs_action = torch.cat([
            obs.view(bs, self.num_nodes, self.obs_dim // self.num_nodes),
            action.view(bs, self.num_nodes, self.action_dim // self.num_nodes),
        ], dim=-1)

        input_features = obs_action.view(bs * self.num_nodes, -1)

        # TODO: Also extact the locations for other time steps
        robot_loc = input_features[:, :self.gnn_action_dim]

        if self.use_time_delays:
            # Removes all the locations from the input features
            aux_s = input_features.shape[0]
            input_features_action = input_features[:, -self.gnn_action_dim:]
            input_features_obs = input_features[:, :-self.gnn_action_dim].reshape(aux_s, self.num_delays, -1)[:, :, self.gnn_action_dim:]
            input_features = torch.cat([
                input_features_obs.reshape(aux_s, -1),
                input_features_action
            ], dim=-1)
        else:
            input_features = input_features[:, self.gnn_action_dim:]

        #batch_idx = torch.arange(bs, device=device).view(-1, 1).repeat(1, self.num_nodes).view(-1)

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
                edges = knn_graph(robot_loc, 5, bs, self.num_nodes)
            elif self.graph_type == 'delaunay':
                edges = delaunay_graph(robot_loc, bs, self.num_nodes)
            elif self.graph_type == 'min_dist':
                r_th = 1.0
                edges = min_distance_graph(robot_loc, bs, self.num_nodes, r_th)
            else:
                raise ValueError(f'Wrong graph type: {self.graph_type}. Provide a value in [formation, complete, knn]')

            #if self.conv_type == 'edge':
            #    edges, _ = add_self_loops(edges, num_nodes=bs * self.num_nodes)

        if self.use_input_mlp:
            input_features1 = self.encoder1(input_features)
            input_features2 = self.encoder2(input_features)
        else:
            input_features1 = input_features
            input_features2 = input_features

        if self.ignore_neighbors:
            # we do not pass the global position to the network
            #net_args = (input_features, )
            net_args1 = (input_features1,)
            net_args2 = (input_features2,)
        else:
            # we do not pass the global position to the network
            # we need to pass the edges when the networks is a GNN

            if self.conv_type == 'node':
                edge_weights = torch.exp(-(robot_loc[edges[1]] - robot_loc[edges[0]]).norm(dim=-1))
                edge_weights = torch.ones_like(edge_weights)
                #net_args = (input_features, edges, edge_weights)
                net_args1 = (input_features1, edges, edge_weights)
                net_args2 = (input_features2, edges, edge_weights)
            elif self.conv_type == 'rel_pos':
                rel_pos = robot_loc[edges[1]] - robot_loc[edges[0]]
                net_args1 = (input_features1, edges, rel_pos)
                net_args2 = (input_features2, edges, rel_pos)
            elif self.conv_type == 'rel_pos_hack':
                rel_pos = robot_loc[edges[1]] - robot_loc[edges[0]]
                net_args1 = (input_features1, edges, rel_pos)
                net_args2 = (input_features2, edges, rel_pos)
            else:
                #net_args = (input_features, edges)
                net_args1 = (input_features1, edges)
                net_args2 = (input_features2, edges)

        if self.residual:
            q1 = self.Q1(*net_args1) + input_features1
            q2 = self.Q2(*net_args2) + input_features2
        else:
            q1 = self.Q1(*net_args1)
            q2 = self.Q2(*net_args2)

        if self.use_output_mlp:
            q1 = self.mlp_Q1(q1)
            q2 = self.mlp_Q1(q2)
        q1 = q1.view(bs, self.num_nodes)
        q2 = q2.view(bs, self.num_nodes)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        netsQ1 = self.Q1 if self.ignore_neighbors else self.Q1.nns
        netsQ2 = self.Q1 if self.ignore_neighbors else self.Q2.nns
        assert len(netsQ1) == len(netsQ2)
        for i, (m1, m2) in enumerate(zip(netsQ1, netsQ2)):
            assert type(m1) == type(m2)
            if type(m1) in [gnn.GCNConv, nn.Linear]:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

        '''
        # For MLP Q nets
        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) in [gnn.GCNConv]:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)'''
        #self.trunk.nns[0].lin.weight.shape
