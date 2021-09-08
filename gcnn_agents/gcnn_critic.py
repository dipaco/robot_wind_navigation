import numpy as np
import torch
import networkx as nx
from torch import nn
from .agent_utils import create_gcnn
from torch_geometric import nn as gnn
from torch_geometric.utils import to_undirected
import torch.nn.functional as F

import utils


class GCNNDoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self,num_nodes, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        assert self.obs_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                                   f' do not divide the observation space size ({self.obs_dim}.)'
        assert self.action_dim % self.num_nodes == 0, f'The number of robots (nodes={self.num_nodes})' \
                                                      f' do not divide the action space size ({self.action_dim}.)'

        # create all the convolutional layers
        gnn_obs_dim = self.obs_dim // self.num_nodes
        gnn_action_dim = self.action_dim // self.num_nodes
        self.Q1 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU)
        self.Q2 = create_gcnn(gnn_obs_dim + gnn_action_dim, 1, self.hidden_dim, self.hidden_depth, non_linearity=nn.ReLU)

        # specify the formation graph
        # TODO: this must come from configuration parameters
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_nodes))
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        self.outputs = dict()
        #self.apply(utils.weight_init)

    def forward(self, obs, action):

        bs = obs.shape[0]

        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([
            obs.view(bs, self.num_nodes, self.obs_dim // self.num_nodes),
            action.view(bs, self.num_nodes, self.action_dim // self.num_nodes)
        ], dim=-1)

        input_features = obs_action.view(bs, self.num_nodes, -1)

        # FIXME: This can be definitely done better using the Batch Class from torch_geometric
        edges = to_undirected(torch.tensor([e for e in self.G.edges], device=input_features.device).long().T)

        q1 = self.Q1(input_features, edges).view(bs, self.num_nodes)
        q2 = self.Q2(input_features, edges).view(bs, self.num_nodes)

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
