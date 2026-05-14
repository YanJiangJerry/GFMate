import pdb
import torch
from config import cfg
from utils.center_embedding import center_embedding_multihop, center_embedding


class Prompt(torch.nn.Module):
    def __init__(
        self, hop_num, label_num, hidden_dim, multi_hop=True, alpha=0.5
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.label_num = label_num
        self.hop_num = hop_num
        self.multi_hop = multi_hop
        ## Weight theta, the learnable prompt for class embedding
        if multi_hop:
            self.weight = torch.nn.Parameter(
                torch.Tensor(hop_num, label_num, hidden_dim)
            )
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(label_num, hidden_dim))
        # self.gamma = self.alpha * torch.pow(
        #     (1 - self.alpha), torch.arange(self.hop_num)
        # )

        ## Fixed weight
        # self.gamma = torch.nn.Parameter(torch.full((hop_num,), 0.5))

        ## Randomly initialize the weight
        # self.gamma = torch.nn.Parameter(torch.Tensor(hop_num))
        # torch.nn.init.uniform_(self.gamma, 0, 1)
        ## Random and the weight sum is 1
        # self.gamma = torch.nn.Parameter(torch.rand(hop_num))
        # self.gamma.data = self.gamma.data / self.gamma.data.sum()

        ## Fix alpha initialisation for all target graphs and learn gamma to generalise cross dataset
        self.alpha = 0.5
        self.gamma = torch.nn.Parameter(
            torch.pow((self.alpha), torch.arange(1, hop_num + 1))
        )
        # self.gamma = torch.nn.Parameter(
        #     torch.pow((self.alpha), torch.arange(hop_num))
        # )
        # import torch.nn.functional as F
        # exponents = torch.pow(self.alpha, torch.arange(self.hop_num))  # [1.0, 0.5, 0.25, ...]
        # self.gamma = torch.nn.Parameter(F.softmax(exponents, dim=0))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, node_embeddings, labels, tuning=True, center=None):
        with torch.no_grad():
            if self.multi_hop:
                centers, class_counts = center_embedding_multihop(
                    input=node_embeddings,
                    index=labels,
                    label_num=self.label_num,
                    hop_num=self.hop_num,
                )
            else:
                centers, class_counts = center_embedding(
                    input=node_embeddings,
                    index=labels,
                    label_num=self.label_num,
                    hop_num=self.hop_num,
                )

        if center is not None:
            return center + self.weight, class_counts
        
        if tuning:
            return centers + self.weight, class_counts
        else:
            return centers, class_counts
