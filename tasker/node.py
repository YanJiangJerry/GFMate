import os
import pdb
import copy
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from config import cfg
import torch.nn.functional as F
from tasker.base import BaseTask
from utils.util import x_padding, x_svd
from utils.data_loader import load4node_gfm
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_edge
from utils.loss import TGCL, OrdinaryLoss, MultiLayerOrdinaryLoss
from utils.seed import set_seed_global
from utils.data_process import split_induced_graphs, node_sample_and_save
from prompts.eval import MultiHopPromptEvaluator, Evaluator, SinglePromptEvaluator


class NodeTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "node"

        self.dataset_name = kwargs["dataset_name"]
        self.load_data_gfm()
        self.generate_few_shot_data()
        # self.initialize_gnn()
        # self.initialize_prompt()
        # self.initialize_optimizer()

    ## Split index by random seed repeat times
    def generate_few_shot_data(self):
        for i in range(1, cfg.repeat + 1):
            cfg.seed = random.randint(0, 100000)
            if cfg.task == "node":
                k_shot_folder = f"./datasets/sample_data/Node/{self.dataset_name}/{self.num_shot}_shot/{i}"
            else:
                k_shot_folder = f"./datasets/sample_data/Graph/{self.dataset_name}/{self.num_shot}_shot/{i}"

            if not os.path.exists(k_shot_folder):
                set_seed_global(cfg.seed)
                os.makedirs(k_shot_folder, exist_ok=True)
                print(f"Sample node for repeat run {i} by random seed {cfg.seed}.")
                node_sample_and_save(
                    self.data, self.num_shot, k_shot_folder, self.output_dim
                )

    ## Obtain the subgraphs for all nodes
    def load_induced_graph(self):
        """
        Split the induced subgraph into training and testing set
        """
        file_path = (
            "./datasets/induced_graph/" + self.dataset_name + "/induced_graph.pkl"
        )
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                graphs_list = pickle.load(f)
        else:
            ## if none, generate the induced graph
            self.data, self.dataset = load4node_gfm(
                self.dataset_name, num_shot=self.num_shot
            )
            self.input_dim = cfg.gfm_dim
            self.output_dim = self.dataset.num_classes
            print("Begin split_induced_graphs and save.")
            split_induced_graphs(
                self.dataset_name, self.data, smallest_size=10, largest_size=30
            )
            with open(file_path, "rb") as f:
                graphs_list = pickle.load(f)
        return graphs_list

    def load_data_gfm(self):
        self.data, self.dataset = load4node_gfm(
            self.dataset_name, num_shot=self.num_shot
        )
        self.data.to(self.device)
        self.input_dim = cfg.gfm_dim
        self.output_dim = self.dataset.num_classes

        if self.data.x.size(-1) < cfg.gfm_dim:
            self.data = x_padding(self.data, cfg.gfm_dim)
        else:
            self.data = x_svd(self.data, cfg.gfm_dim)

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
        ## Scaling the few shot learning
        if test_loader:
            ## Entropy guided selection
            train_loader = self.preprocess_test_loader_entropy(
                test_loader, center_embedding, train_loader
            )
            ## Multi-layer drop edge
            # train_loader = self.preprocess_test_loader_drop_edge(
            #     test_loader, center_embedding, train_loader
            # )

        # train_batch = next(iter(train_loader))
        # for batch in test_loader:
        #     self.optimizer.zero_grad()
        #     batch = batch.to(self.device)
        #     with torch.no_grad():
        #         test_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)
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
            with torch.no_grad():
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
                # print("Using TGCL loss")
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
                # print("Using ordinary loss on the few-shot examples")
                criterion = MultiLayerOrdinaryLoss(self.prompt)
                loss = criterion(
                    out_embeddings,
                    centers,
                    batch.y,
                    center=center_embedding,
                    comp_labels=comp_labels,
                )

            loss.backward()
            # print(centers.requires_grad, self.prompt.weight.requires_grad, self.prompt.weight.grad)
            self.optimizer.step()
            total_loss += loss.item()

        if len(train_loader) > 1:
            print("Test batch number > 1")
            mean_centers = accumulated_centers / (accumulated_counts + 1e-8)
        else:
            mean_centers = centers

        # mean_centers = torch.zeros_like(mean_centers)
        return total_loss / len(train_loader), mean_centers, train_loader

    # def PromptTest(self, train_loader, test_loader=None, center_embedding=None):
    #     # pdb.set_trace()
    #     total_loss = 0.0
    #     accumulated_centers = None
    #     accumulated_counts = None
    #     self.prompt.train()

    #     train_batch = next(iter(train_loader))
    #     for batch in test_loader:
    #         self.optimizer.zero_grad()
    #         batch = batch.to(self.device)
    #         with torch.no_grad():
    #             test_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)

    #             out_embeddings = self.gnn(
    #                 train_batch.x, train_batch.edge_index, train_batch.batch
    #             )

    #         # centers, class_counts = center_embedding_multihop(out_embeddings, batch.y, label_num, self.num_layer + 1)
    #         ## Transfer the center embeddings from the scaled training set
    #         centers, class_counts = self.prompt(out_embeddings, train_batch.y)

    #         # For each class, calculate the average embedding across the batchs
    #         if accumulated_centers is None:
    #             accumulated_centers = centers
    #             accumulated_counts = class_counts
    #         else:
    #             accumulated_centers += centers * class_counts
    #             accumulated_counts += class_counts

    #         criterion = CombinedPromptTuningLoss()
    #         loss = criterion(
    #             out_embeddings,
    #             centers,
    #             train_batch.y,
    #             test_embeddings,
    #             center_embedding,
    #         )

    #         loss.backward()
    #         self.optimizer.step()
    #         total_loss += loss.item()

    #     if len(train_loader) > 1:
    #         print("Test batch number > 1")
    #         mean_centers = accumulated_centers / (accumulated_counts + 1e-8)
    #     else:
    #         mean_centers = centers
    #     return total_loss / len(train_loader), mean_centers

    # def TestTimeTuning(self, test_loader, center, train_loader=None):
    #     self.prompt.train()
    #     total_loss = 0.0
    #     accumulated_centers = None
    #     accumulated_counts = None
    #     for batch in test_loader:
    #         self.optimizer.zero_grad()
    #         batch = batch.to(self.device)

    #         with torch.no_grad():
    #             out_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch)

    #         centers, class_counts = self.prompt(out_embeddings, batch.y, center=center)

    #         if accumulated_centers is None:
    #             accumulated_centers = centers
    #             accumulated_counts = class_counts
    #         else:
    #             accumulated_centers += centers * class_counts
    #             accumulated_counts += class_counts

    #         criterion = TestTimeLoss()
    #         loss = criterion(out_embeddings, centers, batch.y)
    #         # loss = criterion(out_embeddings, centers)
    #         # loss.backward(retain_graph=True)
    #         loss.backward()
    #         self.optimizer.step()
    #         total_loss += loss.item()

    #     if len(test_loader) > 1:
    #         print("Test batch number > 1")
    #         mean_centers = accumulated_centers / (accumulated_counts + 1e-8)
    #     else:
    #         mean_centers = centers
    #     mean_centers = centers
    #     return total_loss / len(test_loader), mean_centers

    ######################################################################################
    def run(self):
        test_accs = []
        ## For testing on different seed data split
        for i in range(1, cfg.repeat + 1):
            print("")
            ## Save the runs for reproducibility
            print(f"Dataset {cfg.dataset.name} Shot {cfg.num_shot} Repeat Runs {i}")

            ## Load data
            # self.initialize_gnn(edge_index_init)
            self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()
            self.gnn.eval()

            ## Get the split index
            if cfg.full_shot:
                try:
                    train_mask = self.dataset[0].train_mask
                    test_mask = self.dataset[0].test_mask

                    if len(train_mask.shape) > 1:
                        train_mask = train_mask[:, 0]
                    if len(test_mask.shape) > 1:
                        test_mask = test_mask[:, 0]

                    train_index = torch.nonzero(train_mask, as_tuple=True)[0]
                    test_index = torch.nonzero(test_mask, as_tuple=True)[0]
                    total_nodes = len(train_mask)
                    train_ratio = len(train_index) / total_nodes
                    test_ratio = len(test_index) / total_nodes
                    print(f"Training data ratio: {train_ratio:.4f}")
                    print(f"Testing data ratio: {test_ratio:.4f}")

                except AttributeError:
                    labels = self.data.y
                    num_labels = labels.max().item() + 1
                    train_index = []
                    test_index = []

                    for label in range(num_labels):
                        label_mask = labels == label
                        label_indices = torch.nonzero(label_mask).squeeze()

                        train_index.append(
                            label_indices[:20]
                        )  # Take 20 nodes for each label as train
                        test_index.append(
                            label_indices[20:]
                        )  # Take the remaining nodes as test

                    train_index = (
                        torch.cat(train_index)
                        if train_index
                        else torch.tensor([], dtype=torch.long)
                    )
                    test_index = (
                        torch.cat(test_index)
                        if test_index
                        else torch.tensor([], dtype=torch.long)
                    )
            else:
                if cfg.task == "node":
                    data_dir = "./datasets/sample_data/Node/{}/{}_shot/{}/".format(
                        self.dataset_name, self.num_shot, i
                    )
                else:
                    data_dir = "./datasets/sample_data/Graph/{}/{}_shot/{}/".format(
                        self.dataset_name, self.num_shot, i
                    )
                train_index = (
                    torch.load(
                        os.path.join(data_dir, "train_idx.pt"), weights_only=False
                    )
                    .type(torch.long)
                    .to(self.device)
                )
                test_index = (
                    torch.load(
                        os.path.join(data_dir, "test_idx.pt"), weights_only=False
                    )
                    .type(torch.long)
                    .to(self.device)
                )

            ## Load all the induced subgraph
            graphs_list = self.load_induced_graph()
            train_graphs, test_graphs = [], []

            num_nodes = num_edges = 0
            labels = set()
            for g in graphs_list:
                if g.index in train_index:
                    train_graphs.append(g)
                elif g.index in test_index:
                    test_graphs.append(g)
                num_nodes += g.x.size(0)
                num_edges += g.edge_index.size(1)
                labels.add(int(g.y))
            if cfg.task == "graph":
                n = len(graphs_list)
                print(
                    f"Graph Dataset & num_graphs {n} & avg_nodes {num_nodes/n:.1f} & avg_edges {num_edges/n:.1f} & num_classes {len(labels)}"
                )

            train_loader = DataLoader(
                train_graphs, batch_size=cfg.optim.batch_size, shuffle=True
            )
            # total_nodes, num_batches = sum(
            #     batch.x.shape[0] for batch in train_loader
            # ), len(train_loader)
            # print(
            #     f"Total number of few shot node (each in subgraph): {total_nodes}, Total number of batches: {num_batches}"
            # )
            # total_nodes, num_batches = 0, 0
            # for batch in train_loader:
            #     num_nodes_in_batch = batch.x.shape[0]
            #     total_nodes += num_nodes_in_batch
            #     num_batches += 1
            #     print(
            #         f"Batch {num_batches}: {num_nodes_in_batch} nodes in all training subgraphs, few shot labels count: {batch.y.shape[0]}"
            #     )

            eval_batch_size = (
                cfg.optim.eval_batch_size
                if cfg.optim.eval_batch_size > 0
                else cfg.optim.batch_size
            )
            test_loader = DataLoader(
                test_graphs, batch_size=eval_batch_size, shuffle=False
            )

            ## Robuestness
            # x_shuffle_ratio = 0.0
            # edge_drop_ratio = 0.0
            # perturbed_test_graphs = []
            # for g in test_graphs:
            #     g = copy.deepcopy(g)
            #     if x_shuffle_ratio > 0:
            #         num_nodes = g.x.size(0)
            #         num_shuffle = int(num_nodes * x_shuffle_ratio)
            #         shuffle_indices = torch.randperm(num_nodes)[:num_shuffle]
            #         shuffled_x = g.x[shuffle_indices[torch.randperm(num_shuffle)]]
            #         g.x[shuffle_indices] = shuffled_x
            #     if edge_drop_ratio > 0:
            #         edge_index, _ = dropout_edge(
            #             g.edge_index,
            #             p=edge_drop_ratio,
            #             force_undirected=False,
            #             training=True,
            #         )
            #         g.edge_index = edge_index
            #     perturbed_test_graphs.append(g)
            # test_loader = DataLoader(
            #     perturbed_test_graphs, batch_size=eval_batch_size, shuffle=False
            # )

            #####################################################################################
            center0 = self.PromptTuning(train_loader, return_centers=True)
            stored_center_embeddings = []
            stored_center_embeddings.append(center0)
            print("Results for pre-training...")
            self.prompt.eval()
            if cfg.multi_hop:
                test_acc = MultiHopPromptEvaluator(
                    test_loader,
                    self.gnn,
                    self.prompt,
                    center0,
                    self.device,
                    self.num_shot,
                    train_loader,
                    self.dataset_name,
                )
            else:
                test_acc = SinglePromptEvaluator(
                    test_loader,
                    self.gnn,
                    self.prompt,
                    center0,
                    self.device,
                    self.num_shot,
                    train_loader,
                    self.dataset_name,
                )

            print(f"Before GFMate: {test_acc:.4f}")
            if cfg.ablation:
                test_accs.append(test_acc)
                continue

            ######################################################################################
            ## Scaling few shot learning
            patience = cfg.optim.patience
            best_loss = 1e9
            cnt_wait = 0
            with tqdm(range(1, self.epochs)) as tq:
                for epoch in tq:
                    ## Tuning layer weights
                    # print(self.prompt.gamma)
                    ## Test time transforming the center embeddings
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
                    # infos = {"epoch": epoch, "loss": loss}
                    # tq.set_postfix(infos)

            if self.epochs == 0:
                center = center0
                stored_center_embeddings.append(center)
            print("Results for Prompt training...")
            self.prompt.eval()
            if cfg.multi_hop:
                test_acc = MultiHopPromptEvaluator(
                    test_loader,
                    self.gnn,
                    self.prompt,
                    center,
                    self.device,
                    self.num_shot,
                    train_loader,
                    self.dataset_name,
                    stored_center_embeddings[0],
                )
            else:
                test_acc = SinglePromptEvaluator(
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

            print(f"After GFMate: {test_acc:.4f}")

            # # ######################################################################################
            # # ## Test time graph prompt tuning
            # # # self.test_performance(test_loader, center)

            # ## The prompt class do not need to be initialised again
            # # self.initialize_prompt()
            # ## Change the learning rate
            # self.initialize_optimizer(self.test_lr)
            # stored_center_embeddings = []
            # stored_center_embeddings.append(center)

            # ## Transform the existing center prompt
            # patience = cfg.optim.patience
            # best_loss = 1e9
            # cnt_wait = 0
            # with tqdm(range(1, self.test_epochs)) as tq:
            #     for epoch in tq:
            #         ## Verify
            #         # loss, centers, scaled_loader = self.PromptTuning(test_loader)

            #         ## Initial center embeddings from prompt phrase
            #         ## Dynamic center embeddings if disabled
            #         # loss, centers = self.TestTimeTuning(test_loader, center, train_loader)

            #         ## Test time tuning
            #         loss, centers = self.PromptTest(scaled_loader, test_loader, center0)

            #         if loss < best_loss:
            #             best_loss = loss
            #             cnt_wait = 0
            #         else:
            #             cnt_wait += 1
            #             if cnt_wait == patience:
            #                 break
            #         infos = {"epoch": epoch, "loss": loss}
            #         tq.set_postfix(infos)

            # ## Test time evaluation
            # # print("After LoRA training...")
            # # print(self.prompt.gamma)
            # print("Results for test-time training...")
            # stored_center_embeddings.append(centers)
            # # print("Is the center prompt the same? ", centers == center)
            # center = centers

            # self.prompt.eval()
            # if cfg.multi_hop:
            #     test_acc = MultiHopPromptEvaluator(
            #         test_loader,
            #         self.gnn,
            #         self.prompt,
            #         center,
            #         self.device,
            #         self.num_shot,
            #         train_loader,
            #         self.dataset_name,
            #         stored_center_embeddings,
            #     )
            # else:
            #     test_acc = SinglePromptEvaluator(
            #         test_loader,
            #         self.gnn,
            #         self.prompt,
            #         center,
            #         self.device,
            #         self.num_shot,
            #         train_loader,
            #         self.dataset_name,
            #     )
            # print(
            #     f"Dataset {cfg.dataset.name} Shot {self.num_shot}: {test_acc:.4f}"
            # )

            print(f"Dataset {cfg.dataset.name}: {test_acc:.2f}")
            if not cfg.ablation:
                test_accs.append(test_acc)
        mean_test_acc = np.mean(test_accs) * 100
        std_test_acc = np.std(test_accs) * 100
        print(f"{mean_test_acc:.2f} | {std_test_acc:.2f}")
        return mean_test_acc
