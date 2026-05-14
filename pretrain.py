import os
import sys
import argparse
from config import cfg
from utils.seed import set_seed_global
from pretrain_strategy.pretraining import Pretraining
import warnings

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

    if len(sys.argv) == 1:
        print("Now you are using the default configs.")
        parser.print_help()

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)

    if cfg.transfer:
        folder_path = f"./ckpts/{cfg.dataset.name}"
    else:
        folder_path = f"./ckpts/{cfg.gfm_dim}_{cfg.optim.pre_train_epochs}"
    if cfg.baseline:
        exit()
    os.makedirs(folder_path, exist_ok=True)
    
    ckpt_path = os.path.join(
        folder_path,
        f"{cfg.model.pretrain_type}-{cfg.model.backbone}-h{cfg.model.hidden_dim}.pth",
    )
    if cfg.debug_pre_train:
        ckpt_path = os.path.join(
            f"./ckpts/test/{cfg.dataset.name}",
            f"{cfg.model.pretrain_type}-{cfg.model.backbone}-h{cfg.model.hidden_dim}.pth",
        )

    if os.path.exists(ckpt_path) and not cfg.debug_pre_train and not cfg.debug:
        # print("Pre-trained model exists, skip pre-training.")
        pass
        
    else:
        set_seed_global(cfg.seed)
        pretrain_mapper = {
            "dagprompt": Pretraining,
        }

        pretrain = pretrain_mapper[cfg.model.pretrain_type](
            gnn_type=cfg.model.backbone,
            dataset_name=cfg.dataset.name,
            hidden_dim=cfg.model.hidden_dim,
            num_layer=cfg.model.num_layers,
            epochs=cfg.optim.pre_train_epochs,
            lr=cfg.optim.pre_train_lr,
            wd=cfg.optim.pre_train_wd,
            device=cfg.device,
        )
        pretrain.pretrain()
