import numpy as np
import torch
import os
import sys
import networkx as nx
from torch import nn
from .agent_utils import create_gcnn, create_mlp, create_output_mlp, zero_weight_init
from torch_geometric import nn as gnn
from torch_geometric.utils import to_undirected, add_self_loops
import torch.nn.functional as F

import utils

__BASE_FOLDER__ = os.path.dirname(os.path.abspath(__file__))
dist_package_folder = os.path.join(__BASE_FOLDER__, '../gcnn_agents/')
sys.path.append(dist_package_folder)
from agent_utils import get_formation_conf

class GCNNDoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, num_nodes, obs_dim, action_dim, hidden_dim, hidden_depth, conv_type, input_batch_norm, formation_type, ignore_neighbors, graph_type):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.conv_type = conv_type
        self.input_batch_norm = input_batch_norm
        self.formation_type = formation_type
        self.use_output_mlp = False
        self.ignore_neighbors = ignore_neighbors
        self.graph_type = graph_type

        assert self.obs_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                                   f' do not divide the observation space size ({self.obs_dim}.)'
        assert self.action_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                                      f' do not divide the action space size ({self.action_dim}.)'

        if self.input_batch_norm:
            self.input_norm_layer_obs = nn.BatchNorm1d(self.obs_dim, track_running_stats=True)
            self.input_norm_layer_action = nn.BatchNorm1d(self.action_dim, track_running_stats=True)

        # create all the convolutional layers
        gnn_obs_dim = self.obs_dim // self.num_nodes
        gnn_action_dim = self.action_dim // self.num_nodes

        if self.use_output_mlp:
            if self.ignore_neighbors:
                self.Q1 = create_mlp(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth)
                self.Q2 = create_mlp(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth)
            else:
                self.Q1 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=False)
                self.Q2 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=False)

            # FIXME: (dipaco) I am unsure if this actually works. I think the dimensions for the output mlp are not right
            self.mlp_Q1 = create_output_mlp(self.hidden_dim, 1)
            self.mlp_Q2 = create_output_mlp(self.hidden_dim, 1)
        else:
            if self.ignore_neighbors:
                self.Q1 = create_mlp(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth)
                self.Q2 = create_mlp(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth)
            else:
                self.Q1 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type)
                self.Q2 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type)

        '''# For MLP Q nets
        self.Q1 = nn.Sequential(
            nn.Linear(gnn_obs_dim + gnn_action_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(gnn_obs_dim + gnn_action_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1, bias=False)
        )'''

        self.G = None
        _, self.G = get_formation_conf(self.formation_type)

        self.outputs = dict()
        #self.apply(zero_weight_init)

    def forward(self, obs, action):

        if self.input_batch_norm:
            # Normalize the input features
            obs = self.input_norm_layer_obs(obs)

            # Normalize the input features
            action = self.input_norm_layer_action(action)

        bs = obs.shape[0]

        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([
            obs.view(bs, self.num_nodes, self.obs_dim // self.num_nodes),
            action.view(bs, self.num_nodes, self.action_dim // self.num_nodes)
        ], dim=-1)

        input_features = obs_action.view(bs * self.num_nodes, -1)

        # FIXME: This can be definitely done better using the Batch Class from torch_geometric
        if self.ignore_neighbors:
            edges = torch.stack(2 * [torch.arange(self.num_nodes, device=input_features.device)]).long()
        else:
            if self.graph_type == 'formation':
                G = self.G
            elif self.graph_type == 'complete':
                G = nx.complete_graph(self.num_nodes)
            elif self.graph_type == 'knn':
                G = nx.complete_graph(self.num_nodes)
                # G.edge[1][2]['weight'] = 3
                # G = nx.k_nearest_neighbors(G, weight='weight')
            else:
                raise ValueError(f'Wrong graph type: {self.graph_type}. Provide a value in [formation, complete, knn]')

            edges = to_undirected(torch.tensor([e for e in G.edges], device=input_features.device).long().T)
            if self.conv_type == 'edge':
                edges, _ = add_self_loops(edges, num_nodes=self.num_nodes)

        edges = torch.cat([edges + i * self.num_nodes for i in range(bs)], dim=-1)

        if self.ignore_neighbors:
            net_args = (input_features, )
        else:
            net_args = (input_features, edges)

        q1 = self.Q1(*net_args)
        q2 = self.Q2(*net_args)
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
