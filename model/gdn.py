import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import GraphAttentionFE

class GDNModel(nn.Module):
    def __init__(self, node_num, edge_indexs, model_cfg, device):
        super(GDNModel, self).__init__()

        self.input_dim = model_cfg["input_dim"]
        self.embedding_dim = model_cfg["embedding_dim"]
        self.out_layer_num = model_cfg["out_layer_num"]
        self.out_layer_inter_dim = model_cfg["out_layer_inter_dim"]
        self.topk = model_cfg["topk"]

        self.edge_indexs = edge_indexs

        self.device = device

        self.embedding = nn.Embedding(node_num, self.embedding_dim)
        self.fe = GraphAttentionFE(self.input_dim, self.embedding_dim)

        self.linear = nn.Linear(self.embedding_dim, 1)

        self.bn_1 = nn.BatchNorm1d(self.embedding_dim)
        self.bn_2 = nn.BatchNorm1d(self.embedding_dim)

        self.dropout = nn.Dropout(0.2)

        self._initialize_parameters()

    def forward(self, x):
        batch_size, node_num, feature_num = x.shape
        edge_num = self.edge_indexs.shape[1]

        node_embeddings = self.embedding(torch.arange(node_num).to(self.device))

        weight_vector = node_embeddings.detach().clone()
        dot_embeddings = torch.mm(weight_vector, weight_vector.T)
        embeddings_norm = weight_vector.norm(dim=-1).reshape(-1, 1)
        node_similarity = dot_embeddings / torch.mm(embeddings_norm, embeddings_norm.T)

        topk_similarity_node_index = torch.topk(node_similarity, self.topk, dim=-1)[1]

        topk_edge_indexs = self._get_topk_edge_indexs(topk_similarity_node_index)
        #topk_edge_indexs = self.edge_indexs.to(self.device)
        batch_edge_index = self._get_batch_edge_index(topk_edge_indexs, batch_size, node_num)

        node_embeddings = node_embeddings.repeat(batch_size, 1)
        x = x.reshape(-1, feature_num)
        y = self.fe(batch_edge_index, x, node_embeddings)
        y = F.relu(self.bn_1(y))
        y = y * node_embeddings
        y = y.reshape(-1, self.embedding_dim)

        y = F.relu(self.bn_2(y))

        y = self.dropout(y)
        y = self.linear(y)
        y = y.reshape(batch_size, node_num)
        return y

    def _get_topk_edge_indexs(self, topk_similarity_node_index):
        node_num = topk_similarity_node_index.shape[0]

        source_nodes = torch.arange(node_num).repeat_interleave(self.topk).to(self.device)
        target_nodes = topk_similarity_node_index.flatten()

        topk_edge_indexs = torch.stack((source_nodes, target_nodes))
        return topk_edge_indexs

    def _get_batch_edge_index(self, edge_indexs, batch_size, node_num):
        edge_num = edge_indexs.shape[1]

        batch_edge_index = edge_indexs.repeat(1, batch_size)
        batch_count_basic = torch.arange(batch_size).repeat_interleave(edge_num).to(self.device)
        batch_count_basic *= node_num
        batch_edge_index += batch_count_basic
        return batch_edge_index

    def _initialize_parameters(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

