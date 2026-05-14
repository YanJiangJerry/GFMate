import os
import sys
import argparse
import warnings
import wandb
import torch
import random
from config import cfg
from tasker.baseline import BaseTask
from tasker.node import NodeTask
from tasker.graph import GraphTask
from utils.seed import set_seed_global
from torch_geometric.data import Data, DataLoader


warnings.filterwarnings("ignore")


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Main entry")
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        default="configs/base.yaml",
        help="Config file path",
        type=str,
    )
    parser.add_argument(
        "--way",
        dest="way",
        default=None,
        help="Number of classes for evaluation; set to 2 to merge test labels into two halves",
        type=int,
    )

    if len(sys.argv) == 1:
        print("Now you are using the default configs.")
        parser.print_help()

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)

    # Optional override for evaluation way (e.g., 2 for binary merge of classes at test time)
    if args.way is not None:
        try:
            cfg.way = int(args.way)
        except Exception:
            # Fallback: keep existing cfg.way if cast fails
            pass

    if cfg.baseline:
        device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        tasker = BaseTask(cfg=cfg, device=device)
        results = tasker.run()
        ## Sweep need to print the results here
        print("Final accuracy:")
        print(results)
        exit()

    ## Force deterministic would increase the gpu memory usage
    set_seed_global(cfg.seed)

    if cfg.wandb:
        wandb.init(project=f"{cfg.dataset.name}_{cfg.num_shot}", config=cfg)
        # cfg.repeat = 1

    if cfg.transfer:
        cfg.pre_train_model_path = os.path.join(
            f"./ckpts/{cfg.dataset.name}",
            f"{cfg.model.pretrain_type}-{cfg.model.backbone}-h{cfg.model.hidden_dim}.pth",
        )
        if cfg.debug_pre_train:
            cfg.pre_train_model_path = os.path.join(
                f"./ckpts/test/{cfg.dataset.name}",
                f"{cfg.model.pretrain_type}-{cfg.model.backbone}-h{cfg.model.hidden_dim}.pth",
            )
    else:
        cfg.pre_train_model_path = os.path.join(
            f"./ckpts/{cfg.gfm_dim}_{cfg.optim.pre_train_epochs}",
            f"{cfg.model.pretrain_type}-{cfg.model.backbone}-h{cfg.model.hidden_dim}.pth",
        )
    ## Minibatch for large datasets
    if cfg.dataset.name == "arxiv-year":
        cfg.repeat = 1

    graph_datasets = ["BZR", "COX2", "ENZYMES", "PROTEINS"]
    if cfg.dataset.name in graph_datasets:
        tasker = GraphTask(
            pre_train_model_path=cfg.pre_train_model_path,
            dataset_name=cfg.dataset.name,
            gnn_type=cfg.model.backbone,
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.hidden_dim,
            prompt_type=cfg.model.prompt_type,
            num_shot=cfg.num_shot,
            epochs=cfg.optim.epochs,
            lr=cfg.optim.lr,
            wd=cfg.optim.wd,
            device=cfg.device,
            r=cfg.model.r,
        )
    else:
        tasker = NodeTask(
            pre_train_model_path=cfg.pre_train_model_path,
            dataset_name=cfg.dataset.name,
            gnn_type=cfg.model.backbone,
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.hidden_dim,
            prompt_type=cfg.model.prompt_type,
            num_shot=cfg.num_shot,
            epochs=cfg.optim.epochs,
            lr=cfg.optim.lr,
            wd=cfg.optim.wd,
            device=cfg.device,
            r=cfg.model.r,
        )

    results = tasker.run()
    ## Sweep need to print the results as the last line
    print("Final accuracy:")
    print(results)

    ## Sweep do not need wandb to log the results here
    if cfg.wandb:
        wandb.log({"accuracy": results})
        wandb.finish()
