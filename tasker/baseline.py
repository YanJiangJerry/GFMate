import os
import pdb
import torch
import numpy as np
from config import cfg
import torch.nn.functional as F
from utils.data_loader import load4node_gfm
from utils.seed import set_seed_global
from model.backbones import GCN, GAT, GraphSAGE


class BaseTask:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.load_data()

    def load_data(self):
        self.data, self.dataset = load4node_gfm(self.cfg.dataset.name, num_shot=None)
        self.data.to(self.device)
        self.input_dim = self.dataset.num_features
        self.output_dim = self.dataset.num_classes

    def build_model(self):
        if self.cfg.baseline_backbone == "GCN":
            model = GCN(
                input_dim=self.input_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                out_dim=self.output_dim,
                num_layer=self.cfg.baseline_layer,
                drop_ratio=cfg.baseline_drop_ratio,
            ).to(self.device)
        elif self.cfg.baseline_backbone == "GAT":
            model = GAT(
                input_dim=self.input_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                out_dim=self.output_dim,
                num_layer=self.cfg.baseline_layer,
                drop_ratio=cfg.baseline_drop_ratio,
            ).to(self.device)
        elif self.cfg.baseline_backbone == "SAGE":
            model = GraphSAGE(
                input_dim=self.input_dim,
                hidden_dim=self.cfg.model.hidden_dim,
                out_dim=self.output_dim,
                num_layer=self.cfg.baseline_layer,
                drop_ratio=cfg.baseline_drop_ratio,
            ).to(self.device)
        else:
            raise ValueError("Invalid model type, choose from ['GCN', 'GAT', 'SAGE']")
        return model

    def train(self, epochs):
        self.model.train()

        try:
            train_mask = self.dataset[0].train_mask
            if len(train_mask.shape) > 1:
                train_mask = train_mask[:, 0]
        except AttributeError:
            labels = self.data.y
            num_labels = labels.max().item() + 1
            train_mask = torch.zeros(labels.size(0), dtype=torch.bool)

            for label in range(num_labels):
                label_mask = labels == label
                label_indices = torch.nonzero(label_mask).squeeze()
                train_indices = label_indices[
                    :cfg.shot_num
                ]  
                train_mask[train_indices] = True

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out_embeddings = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out_embeddings[train_mask], self.data.y[train_mask])
            loss.backward()
            self.optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    def evaluate(self):
        self.model.eval()

        try:
            test_mask = self.dataset[0].test_mask
            if len(test_mask.shape) > 1:
                test_mask = test_mask[:, 0]
        except AttributeError:
            labels = self.data.y
            num_labels = labels.max().item() + 1
            test_mask = torch.zeros(labels.size(0), dtype=torch.bool)

            for label in range(num_labels):
                label_mask = labels == label
                label_indices = torch.nonzero(label_mask).squeeze()
                # Take the remaining nodes as test
                test_indices = label_indices[cfg.shot_num:]  
                test_mask[test_indices] = True
            # test_ratio = test_mask.sum().item() / labels.size(0)
            # print(f"Test ratio: {test_ratio:.2f}")

        with torch.no_grad():
            out_embeddings = self.model(self.data.x, self.data.edge_index)
            pred = out_embeddings.argmax(dim=1)
            correct = (pred[test_mask] == self.data.y[test_mask]).sum().item()
            total = test_mask.sum().item()

        accuracy = correct / total
        return accuracy

    def run(self, repeats=5, epochs=cfg.baseline_epochs):
        accuracies = []
        cfg.shot_num = 20
        for seed in range(repeats):
            set_seed_global(seed)
            self.model = self.build_model()
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params}")
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.baseline_lr,
                weight_decay=self.cfg.baseline_wd,
            )
            self.train(epochs=epochs)
            accuracy = self.evaluate()
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies) * 100
        std_accuracy = np.std(accuracies) * 100
        print(f"acc: {mean_accuracy:.4f} | std: {std_accuracy:.4f}")
        # print(f"Mean Accuracy: {mean_accuracy:.4f}")
        # print(f"Standard Deviation: {std_accuracy:.4f}")
        return mean_accuracy
