import numpy as np
import torch
import os
import sys
import networkx as nx
from torch import nn
from .agent_utils import create_gcnn, create_mlp, zero_weight_init
from torch_geometric import nn as gnn
from torch_geometric.utils import to_undirected
import torch.nn.functional as F

import utils

__BASE_FOLDER__ = os.path.dirname(os.path.abspath(__file__))
dist_package_folder = os.path.join(__BASE_FOLDER__, '../gcnn_agents/')
sys.path.append(dist_package_folder)
from agent_utils import get_formation_conf

class GCNNDoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, num_nodes, obs_dim, action_dim, hidden_dim, hidden_depth, conv_type, input_batch_norm, formation_type):
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
            self.Q1 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=False)
            self.Q2 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU, conv_type=self.conv_type, output_layer=False)

            self.mlp_Q1 = create_mlp(self.hidden_dim, 1)
            self.mlp_Q2 = create_mlp(self.hidden_dim, 1)
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
        edges = to_undirected(torch.tensor([e for e in self.G.edges], device=input_features.device).long().T)
        edges = torch.cat([edges + i*self.num_nodes for i in range(bs)], dim=-1)

        if self.use_output_mlp:
            out1 = self.Q1(input_features, edges)
            out2 = self.Q2(input_features, edges)

            q1 = self.mlp_Q1(out1).view(bs, self.num_nodes)
            q2 = self.mlp_Q1(out2).view(bs, self.num_nodes)
        else:
            q1 = self.Q1(input_features, edges).view(bs, self.num_nodes)
            q2 = self.Q2(input_features, edges).view(bs, self.num_nodes)

        '''# For MLP Q nets
        q1 = self.Q1(input_features).view(bs, self.num_nodes)
        q2 = self.Q2(input_features).view(bs, self.num_nodes)'''

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1.nns) == len(self.Q2.nns)
        for i, (m1, m2) in enumerate(zip(self.Q1.nns, self.Q2.nns)):
            assert type(m1) == type(m2)
            if type(m1) in [gnn.GCNConv]:
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
