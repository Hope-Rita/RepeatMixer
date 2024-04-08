import copy

import numpy as np
import torch
import torch.nn as nn
from audtorch.metrics.functional import pearsonr
from models.modules import TimeEncoder
from utils.utils import NeighborSampler, HistoricalNeighborSampler


class RepeatMixer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 edge_neighbor_sampler: HistoricalNeighborSampler, reflect_table,
                 time_feat_dim: int, num_tokens: int, num_layers: int = 2, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.1, device: str = 'cpu'):
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_tokens: int, number of tokens
        :param num_layers: int, number of transformer layers
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(RepeatMixer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.reflect_table = reflect_table
        self.exist_edges = {}

        self.neighbor_sampler = neighbor_sampler
        self.edge_neighbor_sampler = edge_neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.position_emb = nn.Embedding(2, self.node_feat_dim)
        self.order_emb = nn.Embedding(2, self.node_feat_dim)

        self.time_feat_dim = time_feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.gnn_layers = 2
        self.token_dim_expansion_factor = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout = dropout
        self.device = device

        self.dropout_layer = nn.Dropout(self.dropout)

        self.num_channels = self.edge_feat_dim
        # in GraphMixer, the time encoding function is not trainable
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        self.edge_projection_layer = nn.ModuleList(
            [nn.Linear(self.edge_feat_dim * 3 + time_feat_dim, self.num_channels),
             nn.Linear(self.edge_feat_dim * 3 + time_feat_dim, self.num_channels), ])
        # [nn.Linear(self.edge_feat_dim * 3, self.num_channels),])
        # [nn.Linear(self.edge_feat_dim * 2 + time_feat_dim, self.num_channels),])

        self.edge_mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens * 2, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.cross_mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens * 2, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.concate_layer = nn.Linear(self.num_channels * 2, self.num_channels)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray,
                                                 mask: bool = False,
                                                 num_neighbors: int = 20,
                                                 time_gap: int = 2000):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # # Tensor, shape (batch_size, node_feat_dim)
        edge_embeddings, pcc_one_hop = self.compute_historical_temporal_embeddings(node_src_ids=src_node_ids,
                                                                                   node_dst_ids=dst_node_ids,
                                                                                   node_interact_times=node_interact_times,
                                                                                   mask=mask,
                                                                                   num_neighbors=num_neighbors,
                                                                                   time_gap=time_gap)

        high_order_embbeddings, pcc_two_hop = self.high_order_temporal_embeddings(node_src_ids=src_node_ids,
                                                                                  node_dst_ids=dst_node_ids,
                                                                                  node_interact_times=node_interact_times,
                                                                                  num_neighbors=num_neighbors)
        pcc_index = torch.softmax(torch.stack([pcc_one_hop, pcc_two_hop], dim=-1), dim=-1)
        cat_embeddings = torch.stack([edge_embeddings, high_order_embbeddings], dim=1)
        embeddings = (torch.matmul(pcc_index, cat_embeddings)).squeeze(dim=1)
        return embeddings


    def high_order_temporal_embeddings(self, node_src_ids: np.ndarray,
                                       node_dst_ids: np.ndarray,
                                       node_interact_times: np.ndarray,
                                       num_neighbors: int = 20, ):

        node_src_scopes, neighbor_src_edge_ids, neighbor_src_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_src_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        node_dst_scopes, neighbor_dst_edge_ids, neighbor_dst_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_dst_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        #
        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        src_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_src_edge_ids)]
        dst_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_dst_edge_ids)]

        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        src_neighbor_time_features = self.time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_src_times).float().to(
                self.device))

        dst_neighbor_time_features = self.time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_dst_times).float().to(
                self.device))

        # all_times = np.concatenate([neighbor_src_times, neighbor_dst_times], axis=1).argsort()
        # ndarray, set the time features to all zeros for the padded timestamp
        src_neighbor_time_features[torch.from_numpy(node_src_scopes == 0)] = 0.0
        dst_neighbor_time_features[torch.from_numpy(node_dst_scopes == 0)] = 0.0

        two_hop_src_node_ids, two_hop_src_edge_ids, two_hop_src_times = \
            self.edge_neighbor_sampler.get_two_hop_historical_neighbors(node_ids=node_src_ids, scopes=node_dst_scopes,
                                                                        node_interact_times=node_interact_times,
                                                                        num_neighbors=num_neighbors)

        two_hop_dst_node_ids, two_hop_dst_edge_ids, two_hop_dst_times = \
            self.edge_neighbor_sampler.get_two_hop_historical_neighbors(node_ids=node_dst_ids, scopes=node_src_scopes,
                                                                        node_interact_times=node_interact_times,
                                                                        num_neighbors=num_neighbors)
        two_hop_src_edge_raw_features = self.edge_raw_features[torch.from_numpy(two_hop_src_edge_ids)]
        two_hop_dst_edge_raw_features = self.edge_raw_features[torch.from_numpy(two_hop_dst_edge_ids)]
        src_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - two_hop_src_times).float().to(
            self.device)
        dst_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - two_hop_dst_times).float().to(
            self.device)
        two_hop_src_neighbor_time_features = self.time_encoder(
            timestamps=src_delta_times)
        two_hop_dst_neighbor_time_features = self.time_encoder(
            timestamps=dst_delta_times)
        two_hop_src_neighbor_time_features[torch.from_numpy(two_hop_src_node_ids == 0)] = 0.0
        two_hop_dst_neighbor_time_features[torch.from_numpy(two_hop_dst_node_ids == 0)] = 0.0

        src_combined_features = torch.cat([src_nodes_edge_raw_features, src_neighbor_time_features,
                                           # src_combined_features = torch.cat([src_nodes_edge_raw_features,
                                           self.position_emb(torch.ones_like(  # 1: src
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device))),
                                           self.order_emb(torch.zeros_like(  # 0: one-hop
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device))),
                                           ], dim=-1)

        dst_combined_features = torch.cat([dst_nodes_edge_raw_features, dst_neighbor_time_features,
                                           # dst_combined_features = torch.cat([dst_nodes_edge_raw_features,
                                           self.position_emb(torch.zeros_like(  # 0: dst
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device))),
                                           self.order_emb(torch.zeros_like(  # 0: one-hop
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device)))], dim=-1)

        two_hop_src_combined_features = torch.cat([two_hop_src_edge_raw_features, two_hop_src_neighbor_time_features,
                                                   self.order_emb(torch.ones_like(  # 1: two-hop
                                                       torch.from_numpy(two_hop_src_node_ids).to(self.device))),
                                                   self.position_emb(torch.ones_like(  # src: 1
                                                       torch.from_numpy(two_hop_src_node_ids).to(self.device)))],
                                                  dim=-1)

        two_hop_dst_combined_features = torch.cat([two_hop_dst_edge_raw_features, two_hop_dst_neighbor_time_features,
                                                   self.position_emb(torch.zeros_like(  # 0: dst
                                                       torch.from_numpy(neighbor_src_edge_ids).to(self.device))),
                                                   self.order_emb(torch.ones_like(  # 1: two-hop
                                                       torch.from_numpy(neighbor_src_edge_ids).to(self.device)))],
                                                  dim=-1)

        src_two_hop_combined_features = torch.cat([src_combined_features, two_hop_dst_combined_features], dim=1)
        dst_two_hop_combined_features = torch.cat([dst_combined_features, two_hop_src_combined_features], dim=1)

        src_two_hop_combined_features = self.edge_projection_layer[1](src_two_hop_combined_features)
        dst_two_hop_combined_features = self.edge_projection_layer[1](dst_two_hop_combined_features)

        for mlp_mixer in self.cross_mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            src_two_hop_combined_features = mlp_mixer(src_two_hop_combined_features)
            dst_two_hop_combined_features = mlp_mixer(dst_two_hop_combined_features)

        src_two_hop_combined_features = torch.mean(src_two_hop_combined_features, dim=1)
        dst_two_hop_combined_features = torch.mean(dst_two_hop_combined_features, dim=1)
        #
        src_two_hop_combined_features = torch.nn.functional.sigmoid(src_two_hop_combined_features)
        dst_two_hop_combined_features = torch.nn.functional.tanh(dst_two_hop_combined_features)
        two_hop_combined_features = src_two_hop_combined_features * dst_two_hop_combined_features
        src_var = torch.var(src_delta_times, dim=1)
        dst_var = torch.var(dst_delta_times, dim=1)
        pcc_t = pearsonr(src_delta_times, dst_delta_times)
        pcc_t[src_var == 0] = 0
        pcc_t[dst_var == 0] = 0
        return two_hop_combined_features, pcc_t

    def compute_historical_temporal_embeddings(self, node_src_ids: np.ndarray,
                                               node_dst_ids: np.ndarray,
                                               node_interact_times: np.ndarray,
                                               num_neighbors: int = 20, time_gap: int = 2000, mask: bool = False):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # link encoder
        # get temporal neighbors, including neighbor ids, edge ids and time information
        # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_times, ndarray, shape (batch_size, num_neighbors)

        neighbor_src_node_ids, neighbor_src_edge_ids, neighbor_src_times = \
            self.edge_neighbor_sampler.get_historical_neighbors(node_ids=node_src_ids,
                                                                interact_nodes=node_dst_ids,
                                                                node_interact_times=node_interact_times,
                                                                num_neighbors=num_neighbors)

        neighbor_dst_node_ids, neighbor_dst_edge_ids, neighbor_dst_times = \
            self.edge_neighbor_sampler.get_historical_neighbors(node_ids=node_dst_ids,
                                                                interact_nodes=node_src_ids,
                                                                node_interact_times=node_interact_times,
                                                                num_neighbors=num_neighbors)

        #
        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        src_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_src_edge_ids)]
        dst_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_dst_edge_ids)]
        src_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_src_times).float().to(
            self.device)
        dst_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_dst_times).float().to(
            self.device)

        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        src_neighbor_time_features = self.time_encoder(
            timestamps=src_delta_times)

        dst_neighbor_time_features = self.time_encoder(
            timestamps=dst_delta_times)

        all_times = np.concatenate([neighbor_src_times, neighbor_dst_times], axis=1).argsort()
        # ndarray, set the time features to all zeros for the padded timestamp
        src_neighbor_time_features[torch.from_numpy(neighbor_src_node_ids == 0)] = 0.0
        dst_neighbor_time_features[torch.from_numpy(neighbor_dst_node_ids == 0)] = 0.0

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
        src_combined_features = torch.cat([src_nodes_edge_raw_features, src_neighbor_time_features,
                                           # src_combined_features = torch.cat([src_nodes_edge_raw_features,
                                           self.position_emb(torch.ones_like(  # 1: src
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device))),
                                           self.order_emb(torch.zeros_like(  # 0: one-hop
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device))),
                                           ], dim=-1)

        dst_combined_features = torch.cat([dst_nodes_edge_raw_features, dst_neighbor_time_features,
                                           # dst_combined_features = torch.cat([dst_nodes_edge_raw_features,
                                           self.position_emb(torch.zeros_like(  # 0: dst
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device))),
                                           self.order_emb(torch.zeros_like(  # 0: one-hop
                                               torch.from_numpy(neighbor_src_edge_ids).to(self.device)))], dim=-1)

        combined_features = torch.cat([src_combined_features, dst_combined_features], dim=1)
        combined_features = self.edge_projection_layer[0](combined_features)

        # combined_features = self.edge_projection_layer[0](combined_features)
        # two_hop_combined_features = torch.cat([two_hop_src_combined_features, two_hop_dst_combined_features], dim=1)
        combined_features = combined_features.gather(1, torch.from_numpy(all_times).to(self.device).
                                                     unsqueeze(dim=-1).expand(combined_features.shape[0],
                                                                              combined_features.shape[1],
                                                                              combined_features.shape[2]))

        for mlp_mixer in self.edge_mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            combined_features = mlp_mixer(combined_features)

        # Tensor, shape (batch_size, num_channels)
        combined_features = torch.mean(combined_features, dim=1)
        src_var = torch.var(src_delta_times, dim=1)
        dst_var = torch.var(dst_delta_times, dim=1)
        pcc_t = pearsonr(src_delta_times, dst_delta_times)
        pcc_t[src_var == 0] = 0
        pcc_t[dst_var == 0] = 0

        return combined_features, pcc_t

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def set_edge_neighbor_sampler(self, neighbor_sampler: HistoricalNeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.edge_neighbor_sampler = neighbor_sampler
        if self.edge_neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.edge_neighbor_sampler.seed is not None
            self.edge_neighbor_sampler.reset_random_state()

    def fetch_edge(self, tuple_edges):
        labels = []
        for src_id, dst_id in zip(tuple_edges[0], tuple_edges[1]):
            if (src_id, dst_id) in self.exist_edges:
                labels.append(self.exist_edges[(src_id, dst_id)])
            else:
                labels.append(0)
        return np.array(labels)


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels,
                                                  dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor
