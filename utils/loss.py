import pdb
import torch
import torch.nn as nn
from config import cfg
import torch.nn.functional as F
from geomloss import SamplesLoss


def MultiLinkPredictionLoss(
    node_emb,
    pos_emb,
    neg_emb,
    tau=cfg.optim.pre_train_tau,
    delta=cfg.optim.pre_train_delta,
):
    layers = node_emb.shape[0]
    total_loss = 0
    for l in range(layers):
        x = torch.exp(F.cosine_similarity(node_emb[l], pos_emb[l], dim=-1) / tau)
        y = torch.exp(F.cosine_similarity(node_emb[l], neg_emb[l], dim=-1) / tau)

        pos_prob = x / (x + y)
        neg_prob = y / (x + y)

        info_nce_loss = -torch.log(pos_prob)
        # entropy_loss = -(
        #     pos_prob * torch.log(pos_prob + 1e-8) + neg_prob * torch.log(neg_prob + 1e-8)
        # )
        # loss = delta * entropy_loss + (1 - delta) * info_nce_loss
        loss = info_nce_loss
        total_loss += loss.mean()

    return total_loss


## Conventional CE Loss for Single layer embeddings
class OrdinaryLoss(nn.Module):
    def __init__(self, prompt):
        super(OrdinaryLoss, self).__init__()
        self.tau = cfg.optim.prompt_tau = 0.1
        self.prompt = prompt

    def forward(
        self,
        embeddings,
        center_embeddings,
        labels,
        test_embeddings=None,
        center0=None,
        comp_labels=None,
    ):
        embedding = embeddings[-1]
        center_embedding = center_embeddings[-1]
        similarity_matrix = (
            F.cosine_similarity(
                embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
            )
            * self.prompt.gamma[-1]
            / self.tau
        )  # [N, C]
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


## Conventional CE Loss for Multi-layer embeddings
class MultiLayerOrdinaryLoss(nn.Module):
    def __init__(self, prompt):
        super(MultiLayerOrdinaryLoss, self).__init__()
        self.tau = cfg.optim.prompt_tau = 0.1
        self.prompt = prompt

    def forward(
        self,
        embeddings,
        center_embeddings,
        labels,
        test_embeddings=None,
        center=None,
        comp_labels=None,
    ):
        """
        embeddings: List of [N, D] tensors for each layer
        center_embeddings: List of [C, D] tensors for each layer
        labels: [N] tensor
        """
        total_loss = 0.0
        for i, (embedding, center_embedding) in enumerate(
            zip(embeddings, center_embeddings)
        ):
            similarity_matrix = (
                F.cosine_similarity(
                    embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
                )
                * self.prompt.gamma[i]
                / self.tau
            )  # [N, C]

            loss = F.cross_entropy(similarity_matrix, labels)
            total_loss += loss
        return total_loss


# TGFM Test-time Prompt Tuning Loss
class TGCL(nn.Module):
    def __init__(self, prompt):
        super(TGCL, self).__init__()
        self.delta = cfg.optim.prompt_delta
        self.beta = cfg.optim.prompt_beta
        self.tau = cfg.optim.prompt_tau
        self.prompt = prompt

    def forward(
        self,
        embeddings,
        center_embeddings,
        labels,
        test_embeddings=None,
        center=None,
        comp_labels=None,
    ):

        total_loss = 0
        ordinary_loss = 0
        complementary_loss = 0
        for i, (embedding, center_embedding) in enumerate(
            zip(embeddings, center_embeddings)
        ):
            similarity_matrix = (
                F.cosine_similarity(
                    embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
                )
                * self.prompt.gamma[i]
            )
            exp_similarities = torch.exp(similarity_matrix)
            # probs = exp_similarities / torch.sum(exp_similarities, dim=1, keepdim=True)
            pos = exp_similarities.gather(1, labels.view(-1, 1))
            pos_neg = torch.sum(exp_similarities, dim=1, keepdim=True)
            risk_positive = -torch.log(pos / pos_neg)
            loss_positive = torch.mean(risk_positive)
            if cfg.dataset.name == "citeseer":
                loss_positive = torch.sum(risk_positive)
            ordinary_loss += loss_positive

        ## Minimise the negative class probability for test embeddings
        for i, (embedding, center_embedding) in enumerate(
            zip(test_embeddings, center_embeddings)
        ):
            test_similarity_matrix = (
                F.cosine_similarity(
                    embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
                )
                * self.prompt.gamma[i]
                / self.tau
            )
            test_exp_similarities = torch.exp(test_similarity_matrix)
            ## argmin stop gradient to the negative class, so we pre-compute comp labels
            # comp_labels = test_similarity_matrix.argmin(dim=1, keepdim=True)
            # print(comp_labels, self.delta)

            # log_probs = F.log_softmax(test_similarity_matrix, dim=1)
            # topk_neg = torch.topk(log_probs, k=1, largest=False).indices
            # mask = torch.zeros_like(log_probs).scatter(1, topk_neg, 1.0)
            # loss_negative = -torch.sum(mask * log_probs, dim=1).mean()

            neg = test_exp_similarities.gather(1, comp_labels.view(-1, 1))
            neg_pos = torch.sum(test_exp_similarities, dim=1, keepdim=True)
            risk_negative = -torch.log(1 - neg / neg_pos)
            loss_negative = torch.mean(risk_negative)
            complementary_loss += loss_negative

        ## Entropy loss
        #     for i, (embedding, center_embedding) in enumerate(
        #         zip(test_embeddings, center_embeddings)
        #     ):
        #         test_similarity_matrix = (
        #             F.cosine_similarity(
        #                 embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
        #             )
        #             * self.prompt.gamma[i]
        #             / self.tau
        #         )
        #         probs = F.softmax(test_similarity_matrix, dim=1)
        #         entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        #         loss_entropy = entropy.mean()
        #         complementary_loss += loss_entropy

        ## Soft test loss
        #     for i, (embedding, center_embedding) in enumerate(
        #         zip(test_embeddings, center_embeddings)
        #     ):
        #         sim_matrix = (
        #             F.cosine_similarity(
        #                 embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
        #             )
        #             * self.prompt.gamma[i]
        #             / self.tau
        #         )
        #         log_probs = F.log_softmax(sim_matrix, dim=1)
        #         probs = log_probs.exp()
        #         k = 3
        #         topk_neg = torch.topk(
        #             probs, k=k, largest=False
        #         ).indices  # [num_nodes, k]
        #         mask = torch.zeros_like(probs).scatter(
        #             1, topk_neg, 1.0
        #         )  # [num_nodes, num_classes]
        #         loss_negative = -torch.sum(mask * log_probs, dim=1).mean()
        #         complementary_loss += loss_negative

        total_loss = (1 - self.delta) * ordinary_loss + self.delta * complementary_loss
        return total_loss
