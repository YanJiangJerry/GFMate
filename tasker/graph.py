import os
import pdb
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from config import cfg
from utils.loss import TGCL, OrdinaryLoss, MultiLayerOrdinaryLoss
import torch.nn.functional as F
from tasker.base import BaseTask
from utils.seed import set_seed_global
from utils.util import x_padding, x_svd
from utils.data_loader import load4graph_gfm
from torch_geometric.loader import DataLoader
from utils.data_process import graph_sample_and_save
from prompts.eval import MultiHopPromptEvaluator
from torch_geometric.data import Data, DataLoader


class GraphTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "graph"
        # pdb.set_trace()
        self.dataset_name = kwargs["dataset_name"]
        self.load_data()
        self.answering = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.output_dim), torch.nn.Softmax(dim=1)
        ).to(self.device)

        self.generate_few_shot_data()
        # self.initialize_gnn()
        # self.initialize_prompt()
        # self.initialize_optimizer()
        torch.nn.init.xavier_uniform_(self.answering[0].weight)

    ## Split index by random seed repeat times
    def generate_few_shot_data(self):
        for i in range(1, cfg.repeat + 1):
            cfg.seed = random.randint(0, 100000)
            k_shot_folder = f"./datasets/sample_data/Graph/{self.dataset_name}/{self.num_shot}_shot/{i}"
            if not os.path.exists(k_shot_folder):
                set_seed_global(cfg.seed)
                os.makedirs(k_shot_folder, exist_ok=True)
                print(f"Sample graph for repeat run {i} by random seed {cfg.seed}.")
                graph_sample_and_save(
                    self.dataset, self.num_shot, k_shot_folder, self.output_dim
                )

    def load_data(self):
        self.input_dim, self.output_dim, self.dataset = load4graph_gfm(
            self.dataset_name
        )

    def PromptTuning(
        self,
        train_loader,
        test_loader=None,
        center_embedding=None,
        return_centers=False,
    ):
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None
        if return_centers:
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                with torch.no_grad():
                    out_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)
                centers, class_counts = self.prompt(
                    out_embeddings, batch.y, tuning=False
                )
                # For each class, calculate the average embedding across the batchs
                if accumulated_centers is None:
                    accumulated_centers = centers
                    accumulated_counts = class_counts
                else:
                    accumulated_centers += centers * class_counts
                    accumulated_counts += class_counts
            if len(train_loader) > 1:
                print("Test batch number > 1")
                mean_centers = accumulated_centers / (accumulated_counts + 1e-8)
            else:
                mean_centers = centers
            return mean_centers

        self.prompt.train()
        if test_loader:
            ## Entropy guided selection
            train_loader = self.preprocess_test_loader_entropy(
                test_loader, center_embedding, train_loader
            )
        test_batch = next(iter(test_loader)).to(self.device)
        with torch.no_grad():
            test_embeddings = self.gnn(
                test_batch.x, test_batch.edge_index, test_batch.batch
            )

            layer = cfg.optim.layer
            test_layer_emb = test_embeddings[layer]  # [num_nodes, dim]
            center_emb = center_embedding[layer]  # [num_classes, dim]
            similarity_matrix = F.cosine_similarity(
                test_layer_emb.unsqueeze(1),  # [num_nodes, 1, dim]
                center_emb.unsqueeze(0),  # [1, num_classes, dim]
                dim=-1,  # -> [num_nodes, num_classes]
            )
            comp_labels = similarity_matrix.argmin(dim=1)

        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            out_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)
            centers, class_counts = self.prompt(out_embeddings, batch.y)
            # For each class, calculate the average embedding across the batchs
            if accumulated_centers is None:
                accumulated_centers = centers
                accumulated_counts = class_counts
            else:
                accumulated_centers += centers * class_counts
                accumulated_counts += class_counts

            if cfg.optim.prompt_delta > 0:
                criterion = TGCL(self.prompt)
                loss = criterion(
                    out_embeddings,
                    centers,
                    batch.y,
                    test_embeddings=test_embeddings,
                    center=center_embedding,
                    comp_labels=comp_labels,
                )
            else:
                criterion = MultiLayerOrdinaryLoss(self.prompt)
                loss = criterion(
                    out_embeddings,
                    centers,
                    batch.y,
                    center=center_embedding,
                    comp_labels=comp_labels,
                )

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if len(train_loader) > 1:
            print("Test batch number > 1")
            mean_centers = accumulated_centers / (accumulated_counts + 1e-8)
        else:
            mean_centers = centers
        return total_loss / len(train_loader), mean_centers, train_loader

    def run(self):
        test_accs = []
        for i in range(1, cfg.repeat + 1):
            print("")
            ## Save the runs for reproducibility
            print(f"Dataset {cfg.dataset.name} Shot {cfg.num_shot} Repeat Runs {i}")

            self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()
            self.prompt.eval()
            self.gnn.eval()

            data_dir = "./datasets/sample_data/Graph/{}/{}_shot/{}/".format(
                self.dataset_name, self.num_shot, i
            )
            train_index = (
                torch.load(os.path.join(data_dir, "train_idx.pt"), weights_only=False)
                .type(torch.long)
                .to(self.device)
            )
            test_index = (
                torch.load(os.path.join(data_dir, "test_idx.pt"), weights_only=False)
                .type(torch.long)
                .to(self.device)
            )
            train_dataset = []
            for i in train_index:
                data = self.dataset[i].clone()
                data.y = data.y.squeeze(0)
                data.index = torch.tensor(i)
                train_dataset.append(data)
            test_dataset = [self.dataset[i] for i in test_index]
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            train_loader = DataLoader(
                train_dataset, batch_size=cfg.optim.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=cfg.optim.batch_size, shuffle=False
            )
            center0 = self.PromptTuning(train_loader, return_centers=True)
            test_acc = MultiHopPromptEvaluator(
                test_loader,
                self.gnn,
                self.prompt,
                center0,
                self.device,
                self.num_shot,
                train_loader,
                self.dataset_name,
                # stored_center_embeddings,
            )
            print("Before GFMate: {:.2f} ".format(test_acc * 100))

            patience = cfg.optim.patience
            best_loss = 1e9
            cnt_wait = 0
            with tqdm(range(1, self.epochs)) as tq:
                for epoch in tq:
                    # loss, center = self.PromptTuning(train_loader)
                    loss, center, scaled_loader = self.PromptTuning(
                        train_loader, test_loader, center0
                    )

                    if loss < best_loss:
                        best_loss = loss
                        cnt_wait = 0
                    else:
                        cnt_wait += 1
                        if cnt_wait == patience:
                            break
                    infos = {"epoch": epoch, "loss": loss}
                    tq.set_postfix(infos)

            if self.epochs == 0:
                center = center0
            test_acc = MultiHopPromptEvaluator(
                test_loader,
                self.gnn,
                self.prompt,
                center,
                self.device,
                self.num_shot,
                train_loader,
                self.dataset_name,
                # stored_center_embeddings,
            )
            print("After GFMate: {:.2f} ".format(test_acc * 100))
            test_accs.append(test_acc)

        mean_test_acc = np.mean(test_accs) * 100
        std_test_acc = np.std(test_accs) * 100
        print(f"{mean_test_acc:.2f} | {std_test_acc:.2f}")
        return mean_test_acc
