import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax

class GraphAttentionFE(MessagePassing):
    def __init__(self, in_channels, out_channesl):
        super(GraphAttentionFE, self).__init__()
        self.linear = nn.Linear(in_channels, out_channesl)

        self.attention_vector = nn.Linear((out_channesl+out_channesl)*2, 1)

    def forward(self, edge_indexs, x, node_embeddings):
        """
        Args:
            x (batch_size*node_num, feature_num)
            node_embeddings (node_num, embedding_dim)
        """
        x = self.linear(x)

        edge_indexs, _ = add_self_loops(edge_indexs, num_nodes=x.size(0))

        out = self.propagate(edge_indexs, x=x, node_embeddings=node_embeddings)
        return out


    def message(self, x_i, x_j, edge_index_i, edge_index_j, node_embeddings):
        node_embedding_i = node_embeddings[edge_index_i]
        node_embedding_j = node_embeddings[edge_index_j]

        g_i = torch.cat((x_i, node_embedding_i), dim=-1)
        g_j = torch.cat((x_j, node_embedding_j), dim=-1)
        g = torch.cat((g_i, g_j), dim=-1)

        alpha = self.attention_vector(g)
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index=edge_index_i)

        attention_x_j = alpha * x_j
        return attention_x_j
