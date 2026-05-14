import torch
import torch.nn as nn
from config import cfg
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

## LoRA from DAGPrompt
# from model.layers import GCNConvLoRA, GATConvLoRA
class BaseGNN(torch.nn.Module):
    def __init__(
        self,
        GraphConv,
        input_dim,
        hidden_dim=None,
        out_dim=None,
        num_layer=3,
        JK="last",
        drop_ratio=0.5,
        pool="mean",
    ):
        super().__init__()
        """
        Args:
            num_layer (int): the number of GNN layers
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            pool (str): sum, mean, max, attention, set2set
            
        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
        """

        if hidden_dim is None:
            hidden_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hidden_dim

        ## Projection for input features by a mlp follows DAGPrompt
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        if num_layer < 2:
            raise ValueError(
                "GNN layer_num should >=2 but you set {}".format(num_layer)
            )
        elif num_layer == 2:
            self.layers = torch.nn.ModuleList(
                [GraphConv(hidden_dim, hidden_dim), GraphConv(hidden_dim, out_dim)]
            )
        else:
            layers = [GraphConv(hidden_dim, hidden_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hidden_dim, hidden_dim))
            layers.append(GraphConv(hidden_dim, out_dim))
            self.layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio

        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        # Default is mean pooling
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        self.act = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.input_proj(x)
        h_list = [x]
        for idx, conv in enumerate(self.layers[0:-1]):
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.layers[-1](x, edge_index)
        h_list.append(x)

        ## Default is the last layer node embedding
        if self.JK == "last":
            node_emb = h_list[-1]
        # elif self.JK == "sum":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        # For pre-train without subgraph, directly returns the node embedding
        if batch == None:
            if cfg.baseline:
                return node_emb
            else:
                return h_list

        else:
            ## For downstream tasks without multihop
            graph_embeddings = [
                self.pool(node_embedding, batch.long()) for node_embedding in h_list
            ]
            return torch.stack(graph_embeddings).to(node_emb.device)

    # def forward_res(self, x, edge_index, batch=None):
    #     x = self.input_proj(x)
    #     h_list = [x]

    #     for idx, conv in enumerate(self.layers[0:-1]):
    #         x = conv(x, edge_index)
    #         x = self.act(x)
    #         x = F.dropout(x, self.drop_ratio, training=self.training)
    #         h_list.append(x)

    #     x = self.layers[-1](x, edge_index)
    #     h_list.append(x)

    #     ## residual graph embeddings
    #     residual_h_list = [h_list[0]]
    #     # residual_h_list = []
    #     for i in range(1, len(h_list)):  # residual
    #         residual_h_list.append(h_list[i] - h_list[i - 1])
    #     # residual_h_list.append(h_list[-1])

    #     if batch == None:
    #         return residual_h_list[-1]
    #     else:
    #         graph_embeddings = [
    #             self.pool(node_embedding, batch.long()) for node_embedding in residual_h_list
    #         ]
    #         return torch.stack(graph_embeddings).to(x.device)


class GCN(BaseGNN):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        out_dim=None,
        num_layer=3,
        JK="last",
        drop_ratio=0,
        pool="mean",
    ):
        super().__init__(
            GCNConv, input_dim, hidden_dim, out_dim, num_layer, JK, drop_ratio, pool
        )


class GAT(BaseGNN):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        out_dim=None,
        num_layer=3,
        JK="last",
        drop_ratio=0,
        pool="mean",
    ):
        super().__init__(
            GATConv, input_dim, hidden_dim, out_dim, num_layer, JK, drop_ratio, pool
        )


class GraphSAGE(BaseGNN):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        out_dim=None,
        num_layer=3,
        JK="last",
        drop_ratio=0,
        pool="mean",
    ):
        super().__init__(
            SAGEConv, input_dim, hidden_dim, out_dim, num_layer, JK, drop_ratio, pool
        )
