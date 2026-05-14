import pdb
import torch
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from config import cfg
from utils.util import x_padding, x_svd
from pretrain_strategy.base import PreTrainBase
from utils.loss import (
    # LinkPredictionLoss,
    MultiLinkPredictionLoss,
)
from utils.data_process import edge_index_to_sparse_matrix, prepare_structured_data
from utils.data_loader import (
    load4link_prediction_single_graph,
    load4link_prediction_multi_graph,
)


class Pretraining(PreTrainBase):
    """
    Pretraining strategy (edge prediction) for Distribution-aware Graph Prompting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_type = cfg.model.pretrain_type
        ## GFM
        self.dataloader = self.generate_loader_data_gfm()
        self.input_dim = cfg.gfm_dim
        self.initialize_gnn(self.input_dim, self.hidden_dim)
        # self.fc = Linear(self.hidden_dim, self.output_dim).to(self.device)
        self.initialize_optimizer()

    def initialize_optimizer(self):
        # parameters_group = list(self.gnn.parameters()) + list(self.fc.parameters())
        parameters_group = list(self.gnn.parameters())
        self.optimizer = torch.optim.Adam(
            parameters_group, lr=self.lr, weight_decay=self.wd
        )

    def generate_loader_data_gfm(self):
        graph_datasets = [
            ## Node Classification
            "citeseer",
            "cora",
            "photo",
            "texas",
            "wisconsin",
            "cornell",
            "chameleon",
            "squirrel",
            ## May OOM if not enough memory
            # "arxiv-year",
            # "reddit",
            # "wikics",
            # "flickr",
            ## Graph Classification
            # "MUTAG",
            # "COX2",
            # "ENZYMES",
            # "COLLAB",
            # "PROTEINS",
            # "IMDB-BINARY",
            # "REDDIT-BINARY",
            # "BZR",
            # "PTC_MR",
        ]

        if cfg.task == "node":
            load_function = load4link_prediction_single_graph
        elif cfg.task == "graph":
            graph_datasets = [
                ## Node Classification
                "citeseer",
                "cora",
                "photo",
                "texas",
                "wisconsin",
                "cornell",
                "chameleon",
                ## Graph Classification
                "BZR",
                "COX2",
                "ENZYMES",
                "PROTEINS",
                ## Too small
                # "MUTAG",
                # "PTC_MR",
                ## Too large, OOM
                # "arxiv-year",
                # "squirrel",
                ## Non-feature
                # "COLLAB",
                # "IMDB-BINARY",
                # "REDDIT-BINARY",
            ]
            load_function = load4link_prediction_multi_graph
        else:
            raise NotImplementedError("invalid dataset name")

        all_edge_index = []
        all_features = []
        all_node_ids = []
        total_nodes = 0
        for dataset_name in graph_datasets:
            ## Transfer learning setting, pretrain on other datasets
            if cfg.transfer and dataset_name == self.dataset_name:
                continue

            data, edge_label, edge_index, input_dim, self.output_dim = load_function(
                dataset_name
            )

            if data.x.size(-1) < cfg.gfm_dim:
                data = x_padding(data, cfg.gfm_dim)
            elif data.x.size(-1) > cfg.gfm_dim:
                data = x_svd(data, cfg.gfm_dim)

            ## Add new node ids
            num_nodes = data.x.size(0)
            new_ids = torch.arange(total_nodes, total_nodes + num_nodes)
            all_node_ids.append(new_ids)

            ## Re-index edge_index
            all_edge_index.append(data.edge_index + total_nodes)
            all_features.append(data.x)
            total_nodes += num_nodes

        print(f"Total pre-training nodes except target datasets: {total_nodes}")
        final_edge_index = torch.cat(all_edge_index, dim=-1)
        final_features = torch.cat(all_features, dim=0)
        self.data = Data(x=final_features, edge_index=final_edge_index)
        # self.adj = edge_index_to_sparse_matrix(
        #     self.data.edge_index, self.data.x.shape[0]
        # ).to(self.device)

        ## Prepare link prediction pre-training data
        data = prepare_structured_data(self.data)
        return DataLoader(
            TensorDataset(data), batch_size=cfg.optim.pre_train_batch_size, shuffle=True
        )

    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        self.gnn.train()
        for _, batch in enumerate(self.dataloader):
            # pdb.set_trace()
            self.optimizer.zero_grad()
            batch = batch[0]
            batch = batch.to(self.device)

            out = self.gnn(
                self.data.x.to(self.device), self.data.edge_index.to(self.device)
            )

            all_node_emb = torch.stack(out)
            node_emb = all_node_emb[:, batch[:, 0], :]
            pos_emb = all_node_emb[:, batch[:, 1], :]
            neg_emb = all_node_emb[:, batch[:, 2], :]
            loss = MultiLinkPredictionLoss(node_emb, pos_emb, neg_emb)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1

        return accum_loss / total_step