import logging
import os
from yacs.config import CfgNode as CN

# Global config object
cfg = CN()


def set_cfg(cfg):
    r"""
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name
    """
    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Select the device, cpu or cuda
    cfg.device = "cuda:0"
    cfg.wandb = False
    ## Follow DAGPrompt for consistent and fair comparison
    ## DAGPrompt fixed seed for 5 different runs
    cfg.seed = 1
    cfg.repeat = 5
    cfg.model_seed = cfg.seed
    # Num labels per class for few-shot learning
    cfg.num_shot = 10
    # Ratio of nodes for validation and hyperparameter tuning
    cfg.val_ratio = 0.2
    # Path to the pre-trained model, only effective for downstream tasks
    cfg.pre_train_model_path = None

    # Evaluation class merging (e.g., 2 for binary by merging first/second half of classes at test time)
    # Default None means keep original class granularity
    cfg.way = None

    # Node classification, or graph classification task?
    cfg.task = "node"  # node or graph

    ## Compare with ordinary method predict by lasy layer
    cfg.last_layer = False

    ## Ablation study
    cfg.ablation = False

    ## Base GNN
    cfg.baseline = False
    cfg.baseline_backbone = "SAGE"
    cfg.baseline_layer = 3
    cfg.baseline_epochs = 200
    cfg.baseline_lr = 1e-3
    cfg.baseline_wd = 0.0
    cfg.baseline_drop_ratio = 0.5

    ## Full shot
    cfg.full_shot = False

    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #
    cfg.dataset = CN()
    cfg.dataset.name = "texas"
    # Modified automatically by code, no need to set
    cfg.dataset.num_nodes = -1
    # Modified automatically by code, no need to set
    cfg.dataset.num_classes = -1
    # Dir to load the dataset. If the dataset is downloaded, it is in root
    cfg.dataset.root = "../datasets"

    # ------------------------------------------------------------------------ #
    # Optimization options
    # ------------------------------------------------------------------------ #
    cfg.optim = CN()

    ## Pre-training
    cfg.transfer = True
    cfg.debug = False
    cfg.debug_pre_train = False
    cfg.optim.pre_train_delta = 0.1
    cfg.optim.pre_train_lr = 1e-4
    cfg.optim.pre_train_wd = 0.0
    cfg.optim.pre_train_epochs = 10
    cfg.gfm_dim = 100

    ## Prompt training at test-time
    cfg.optim.lr = 1e-4
    ## Scaling number
    cfg.optim.test_shot = 1
    cfg.optim.epochs = 10
    cfg.optim.patience = 50
    ## The weight for complementary loss
    cfg.optim.prompt_delta = 0.1
    cfg.optim.prompt_beta = 1.0
    cfg.optim.prompt_tau = 0.1
    cfg.optim.wd = 5e-4

    ## Class
    cfg.optim.k = 1
    cfg.optim.repeat = 1

    ## Temperature default 0.1
    cfg.optim.pre_train_tau = 0.1
    cfg.optim.test_tau = 0.1
    cfg.optim.prompt_tau = 0.1

    ## Multi-layer control
    ## The pivot layer is automatically selected by the entropy
    cfg.optim.layer = None
    cfg.optim.mean_layer = False
    cfg.optim.layer_consistent = False

    # Batch size
    cfg.optim.pre_train_batch_size = cfg.optim.batch_size = 1000000
    cfg.optim.eval_batch_size = -1

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    ## Default layer weight 0.5
    cfg.model.alpha = 0.5
    # Backbone model to use
    cfg.model.backbone = "GCN"
    # Prompt type, we eliminate the effect of dagprompt prompt in the method
    cfg.model.pretrain_type = "linkpred"
    cfg.model.prompt_type = "gfmate"

    # Hidden layer dim
    cfg.model.hidden_dim = 256
    # Layer number
    cfg.model.num_layers = 3
    # Dropout rate
    cfg.model.dropout = 0.5
    # Pooling method, sum, mean, max
    cfg.model.pool = "mean"
    # JK method: how the node features across layers are combined. last, sum, max or concat
    cfg.model.JK = "last"
    cfg.model.r = 0
    ## DAGPrompt tune gamma and node embedding weight for node prompt
    cfg.model.lg = False
    cfg.model.adaptive_adj = False


set_cfg(cfg)
