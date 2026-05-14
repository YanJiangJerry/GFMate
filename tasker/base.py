import pdb
import torch
from config import cfg
from torch.optim import Adam
import torch.nn.functional as F
from prompts.prompt import Prompt
from model.backbones import GCN, GAT, GraphSAGE
from torch_geometric.data import Data, DataLoader


## Backbone model
class BaseTask:
    def __init__(
        self,
        pre_train_model_path,
        dataset_name,
        gnn_type="GCN",
        num_layers=2,
        hidden_dim=128,
        prompt_type="none",
        num_shot=10,
        epochs=100,
        lr=1e-3,
        wd=5e-6,
        device=1,
        r=0,
    ):
        self.pre_train_model_path = pre_train_model_path
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = num_layers
        self.hidden_dim = hidden_dim
        self.prompt_type = prompt_type
        self.num_shot = num_shot
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.device = torch.device(device)
        self.r = r
        self.initialize_lossfn()
        ## Default multi-hop prompt
        cfg.multi_hop = True
        # self.test_epochs = cfg.optim.test_epochs
        # self.test_lr = cfg.optim.test_lr


    def initialize_optimizer(self):
        param_group = []
        param_group.append({"params": self.prompt.parameters()})
        self.optimizer = Adam(param_group, lr=self.lr, weight_decay=self.wd)

        # Calculate the total number of tunable parameters
        total_tunable_params = 0
        for param in param_group:
            for p in param["params"]:
                total_tunable_params += sum(p.numel() for p in p if p.requires_grad)
        print(f"Total tunable parameters: {total_tunable_params}")
        # exit()

    def initialize_lossfn(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def initialize_prompt(self):
        ## Add node feature itself num_layers = num_hops + 1
        self.prompt = Prompt(
            hop_num=self.num_layer + 1,
            label_num=self.output_dim,
            hidden_dim=self.hidden_dim,
            multi_hop=cfg.multi_hop,
            alpha=cfg.model.alpha,
        ).to(self.device)

    def initialize_gnn(self, edge_index=None):
        # pdb.set_trace()
        ## GFM
        self.input_dim = cfg.gfm_dim
        if self.gnn_type == "GAT":
            self.gnn = GAT(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layer=self.num_layer,
            )

        elif self.gnn_type == "GCN":
            self.gnn = GCN(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layer=self.num_layer,
            )

        elif self.gnn_type == "SAGE":
            self.gnn = GraphSAGE(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layer=self.num_layer,
            )
        else:
            raise ValueError(f"Unsupport GNN: {self.gnn_type}")
        self.gnn.to(self.device)
        result = self.gnn.load_state_dict(
            torch.load(
                self.pre_train_model_path,
                map_location=self.device,
                weights_only=False,
            ),
            strict=False,
        )
        print("Pre-train GFM:", result)

    def preprocess_test_loader_entropy(
        self, test_loader, center_embeddings, train_loader=None
    ):
        if cfg.optim.test_shot == 0:
            return train_loader

        selected_indices = []
        all_data_list = []
        all_preds = []
        all_entropies = []

        for batch in test_loader:
            batch = batch.to(self.device)
            out_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)

            ## Find the pivot layer
            if cfg.optim.layer == None:
                best_entropy = float("inf")
                best_layer_entropy = None
                best_pred = None
                best_layer = 0
                for i, (out_embedding, center_embedding) in enumerate(
                    zip(out_embeddings, center_embeddings)
                ):
                    similarity_matrix = (
                        F.cosine_similarity(
                            out_embedding.unsqueeze(1),
                            center_embedding.unsqueeze(0),
                            dim=-1,
                        )
                        / cfg.optim.prompt_tau
                    )
                    exp_similarities = torch.exp(similarity_matrix)
                    probs = exp_similarities / torch.sum(
                        exp_similarities, dim=1, keepdim=True
                    )
                    pred = probs.argmax(dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    mean_entropy = entropy.mean().item()
                    if mean_entropy < best_entropy:
                        best_entropy = mean_entropy
                        best_layer_entropy = entropy
                        best_pred = pred
                        best_layer = i
                cfg.optim.layer = best_layer
                # print(f"Best layer: {best_layer}, entropy: {best_entropy}")
                # pred, entropy = best_pred, best_layer_entropy

            ## Accumulated similarity
            if cfg.optim.mean_layer:
                accumulated_similarity_matrix = torch.zeros(
                    out_embeddings.size(1), center_embeddings.size(1)
                ).to(self.device)
                for i, (out_embedding, center_embedding) in enumerate(
                    zip(out_embeddings, center_embeddings)
                ):
                    similarity_matrix = (
                        F.cosine_similarity(
                            out_embedding.unsqueeze(1),
                            center_embedding.unsqueeze(0),
                            dim=-1,
                        )
                        # / cfg.optim.prompt_tau
                    )
                    # accumulated_similarity_matrix += similarity_matrix * self.prompt.gamma[i]
                    accumulated_similarity_matrix += similarity_matrix
                exp_similarities = torch.exp(accumulated_similarity_matrix)
                probs = exp_similarities / torch.sum(
                    exp_similarities, dim=1, keepdim=True
                )
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                pred = accumulated_similarity_matrix.argmax(dim=1)

            elif cfg.optim.layer_consistent:
                ## Consistent layer similarity
                accumulated_similarity_matrices = []
                predictions_per_layer = []

                for i, (out_embedding, center_embedding) in enumerate(
                    zip(out_embeddings, center_embeddings)
                ):
                    similarity_matrix = F.cosine_similarity(
                        out_embedding.unsqueeze(1),
                        center_embedding.unsqueeze(0),
                        dim=-1,
                    )
                    accumulated_similarity_matrices.append(similarity_matrix)
                    predictions_per_layer.append(similarity_matrix.argmax(dim=1))

                predictions_per_layer = torch.stack(predictions_per_layer, dim=0)
                # consistent_mask = (predictions_per_layer == predictions_per_layer[0]).all(dim=0)
                consistent_mask = (
                    predictions_per_layer == predictions_per_layer[cfg.optim.layer]
                ).all(dim=0)
                ## Invert the mask
                # consistent_mask = ~consistent_mask

                # accumulated_similarity_matrix = torch.stack(accumulated_similarity_matrices, dim=0).sum(dim=0)
                accumulated_similarity_matrix = accumulated_similarity_matrices[
                    cfg.optim.layer
                ]

                accumulated_similarity_matrix = accumulated_similarity_matrix[
                    consistent_mask
                ]
                exp_similarities = torch.exp(accumulated_similarity_matrix)

                pred = predictions_per_layer[cfg.optim.layer][consistent_mask]
                probs = exp_similarities / torch.sum(
                    exp_similarities, dim=1, keepdim=True
                )
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            else:
                out_embeddings = out_embeddings[cfg.optim.layer]
                center_embedding = center_embeddings[cfg.optim.layer]
                similarity_matrix = (
                    F.cosine_similarity(
                        out_embeddings.unsqueeze(1),
                        center_embedding.unsqueeze(0),
                        dim=-1,
                    )
                    / cfg.optim.prompt_tau
                )
                exp_similarities = torch.exp(similarity_matrix)
                probs = exp_similarities / torch.sum(
                    exp_similarities, dim=1, keepdim=True
                )
                pred = probs.argmax(dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

                ## Select and then consistent layer similarity
                # selected_layer = cfg.optim.layer
                # selected_out_embedding = out_embeddings[selected_layer]
                # selected_center_embedding = center_embeddings[selected_layer]
                # similarity_matrix = (
                #     F.cosine_similarity(
                #         selected_out_embedding.unsqueeze(1),
                #         selected_center_embedding.unsqueeze(0),
                #         dim=-1,
                #     )
                #     / cfg.optim.prompt_tau
                # )
                # pred = similarity_matrix.argmax(dim=1)
                # predictions_per_layer = []

                # for i, (out_embedding, center_embedding) in enumerate(zip(out_embeddings, center_embeddings)):
                #     similarity_matrix_2 = F.cosine_similarity(
                #         out_embedding.unsqueeze(1),
                #         center_embedding.unsqueeze(0),
                #         dim=-1,
                #     )
                #     predictions_per_layer.append(similarity_matrix_2.argmax(dim=1))

                # predictions_per_layer = torch.stack(predictions_per_layer, dim=0)
                # consistent_mask = (predictions_per_layer == pred).all(dim=0)
                # consistent_mask = ~consistent_mask

                # similarity_matrix = similarity_matrix[consistent_mask]
                # pred = pred[consistent_mask]

                # exp_similarities = torch.exp(similarity_matrix)
                # probs = exp_similarities / torch.sum(exp_similarities, dim=1, keepdim=True)
                # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            ## Save the data for each batch
            all_data_list.extend(batch.to_data_list())
            all_preds.append(pred)
            all_entropies.append(entropy)

        all_preds = torch.cat(all_preds, dim=0)
        all_entropies = torch.cat(all_entropies, dim=0)
        all_y = torch.cat([data.y for data in all_data_list], dim=0)

        for class_id in all_y.unique():
            class_mask = all_preds == class_id
            class_indices = class_mask.nonzero(as_tuple=True)[0]
            if class_mask.sum() > 0:
                class_entropy = all_entropies[class_mask]
                num_select = min(cfg.optim.test_shot, class_entropy.shape[0])
                lowest_entropy_indices = class_entropy.topk(
                    num_select, largest=False
                ).indices
                selected_indices.append(class_indices[lowest_entropy_indices])

        final_selected = torch.cat(selected_indices, dim=0)

        ## Change label to the predicted label as the groundtruth label is not accessible at test time
        new_batch = []
        for index in final_selected:
            data = all_data_list[index]
            data.y = torch.tensor(
                all_preds[index], dtype=torch.long, device=self.device
            )
            data.index = torch.tensor(index, dtype=torch.long, device=self.device)
            new_batch.append(data)

        if train_loader is not None:
            new_batch.extend(train_loader.dataset)

        updated_batch = []
        for data in new_batch:
            x = torch.tensor(data.x, dtype=torch.float).to(self.device)
            y = torch.tensor(data.y, dtype=torch.long).to(self.device)
            index = torch.tensor(data.index, dtype=torch.long).to(self.device)
            updated_batch.append(
                Data(x=x, edge_index=data.edge_index.to(self.device), y=y, index=index)
            )

        updated_loader = DataLoader(
            updated_batch, batch_size=cfg.optim.batch_size, shuffle=True
        )

        return updated_loader

    def preprocess_test_loader_drop_edge(
        self, test_loader, center_embedding, train_loader=None
    ):
        selected_indices = None
        all_batches = []
        all_preds = []
        all_entropies = []

        for batch in test_loader:
            batch = batch.to(self.device)
            saved_edges = batch.edge_index.clone()

            ## Drop edges
            for drop_ratio in torch.arange(0, 1.1, 0.1).tolist():
                if drop_ratio > 0:
                    num_edges = batch.edge_index.shape[1]
                    mask = (
                        torch.rand(num_edges, device=batch.edge_index.device)
                        > drop_ratio
                    )
                    batch.edge_index = batch.edge_index[:, mask]

                out_embeddings_all = self.gnn(batch.x, batch.edge_index, batch.batch)
                num_layers = len(out_embeddings_all)

                preds_all = []
                entropy_all = []

                for i in range(num_layers):
                    similarity_matrix = (
                        F.cosine_similarity(
                            out_embeddings_all[i].unsqueeze(1),
                            center_embedding[i].unsqueeze(0),
                            dim=-1,
                        )
                        / cfg.optim.prompt_tau
                    )
                    exp_similarities = torch.exp(similarity_matrix)
                    probs = exp_similarities / torch.sum(
                        exp_similarities, dim=1, keepdim=True
                    )
                    pred = probs.argmax(dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

                    preds_all.append(pred)
                    entropy_all.append(entropy)

                ## Consistency check
                preds_all = torch.stack(preds_all)
                entropy_all = torch.stack(entropy_all)
                consistent_mask = (preds_all == preds_all[0]).all(dim=0)
                consistent_indices = consistent_mask.nonzero(as_tuple=True)[0]

                if selected_indices is None:
                    selected_indices = set(consistent_indices.tolist())
                else:
                    selected_indices &= set(consistent_indices.tolist())

            all_batches.append(batch)
            all_preds.append(preds_all)
            all_entropies.append(entropy_all)

        if len(self.stored_pivot_nodes) > 0:
            selected_indices = self.stored_pivot_nodes
        else:
            self.stored_pivot_nodes = selected_indices

        selected_indices = torch.tensor(list(selected_indices), device=batch.y.device)
        mean_entropy = torch.cat(
            [ent[:, selected_indices] for ent in all_entropies], dim=0
        ).mean(dim=0)
        sorted_indices = selected_indices[mean_entropy.argsort(descending=False)]

        ## For each class, select the top nodes
        final_selected = []
        for batch, preds in zip(all_batches, all_preds):
            for class_id in batch.y.unique():
                class_mask = preds[0][sorted_indices] == class_id
                class_indices = sorted_indices[class_mask]
                if class_indices.shape[0] > 0:
                    num_select = min(cfg.optim.test_shot, class_indices.shape[0])
                    final_selected.append(class_indices[:num_select])

        ## Final selected nodes
        final_selected = torch.cat(final_selected, dim=0)

        ## Change label to the predicted label as the groundtruth label is not accessible at test time
        new_batch = []
        for batch, preds in zip(all_batches, all_preds):
            batch.edge_index = saved_edges
            for index in final_selected:
                item = batch[index]
                item.y = torch.tensor(preds[0][index], dtype=torch.long).to(self.device)
                item.index = torch.tensor(index, dtype=torch.long).to(self.device)
                new_batch.append(item)

        updated_loader = DataLoader(
            new_batch, batch_size=cfg.optim.batch_size, shuffle=True
        )


        # Concat with the training nodes
        if train_loader is not None:
            new_batch.extend(train_loader.dataset)

            updated_batch = []
            ## Type and device should match the training nodes
            for data in new_batch:
                x = torch.tensor(data.x, dtype=torch.float).to(self.device)
                y = torch.tensor(data.y, dtype=torch.long).to(self.device)
                index = torch.tensor(data.index, dtype=torch.long).to(self.device)
                new_data = Data(
                    x=x, edge_index=data.edge_index.to(self.device), y=y, index=index
                )
                updated_batch.append(new_data)

            updated_loader = DataLoader(
                updated_batch, batch_size=cfg.optim.batch_size, shuffle=True
            )

        return updated_loader
