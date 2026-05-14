import wandb
import yaml
import os
import subprocess
import argparse
import subprocess
import time

# os.environ["WANDB_MODE"] = "offline"
## wandb sync


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Configuration")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name (e.g., cora, amazon, etc.)",
    )
    parser.add_argument(
        "--gnn",
        type=str,
        default="GCN",
        help="Specify the GNN model (e.g., GCN, GAT, SAGE)",
    )
    return parser.parse_args()


def load_base_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sweep_function():
    with wandb.init():
        config = wandb.config

        dataset = config.get("dataset.name", "cora")
        gnn = config.get("gnn", "GCN")
        config_path = f"configs/{dataset}/1-shot/tgfm.yaml"

        base_config = load_base_config(config_path)

        base_config.setdefault("optim", {})
        base_config.setdefault("model", {})
        # base_config.setdefault("seed", 1)
        base_config.setdefault("baseline", True)
        base_config.setdefault("baseline_backbone", gnn)

        # Directly update with sweep parameters
        base_config["baseline_lr"] = float(config.get("baseline_lr", 1e-3))
        base_config["baseline_wd"] = float(config.get("baseline_wd", 0.0))
        base_config["baseline_epochs"] = int(config.get("baseline_epochs", 200))
        base_config["baseline_layer"] = int(config.get("baseline_layer", 3))
        base_config["baseline_drop_ratio"] = float(
            config.get("baseline_drop_ratio", 0.0)
        )

        updated_cfg_path = f"configs/{dataset}/1-shot/gptt.yaml"
        os.makedirs(os.path.dirname(updated_cfg_path), exist_ok=True)
        with open(updated_cfg_path, "w") as f:
            yaml.dump(base_config, f)

        start_time = time.time()
        result = subprocess.run(
            f"CUDA_LAUNCH_BLOCKING=1 python downstream.py --cfg {updated_cfg_path}",
            shell=True,
            check=True,
            text=True,
            capture_output=True,
        )
        end_time = time.time()
        duration = end_time - start_time
        print(f"Finished in {duration:.2f} seconds.")

        second_last_line = result.stdout.strip().split("\n")[-3]
        accuracy = float(result.stdout.strip().split("\n")[-1])
        print(f"Dataset: {dataset} GNN: {gnn}")
        print(f"{second_last_line}")
        wandb.log({"accuracy": accuracy})


def main():
    args = parse_args()
    project_name = f"{args.dataset}_{args.gnn}_base"
    ## random, grid, greedy, bayes
    sweep_config = {
        "method": "bayes",
        "run_cap": 10,
        "name": project_name,
        "project": project_name,
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "baseline_layer": {"values": [2, 3, 4, 5]},
            "baseline_epochs": {"values": [20, 50, 100, 200, 500]},
            "baseline_lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
            "baseline_wd": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
            "baseline_drop_ratio": {"values": [0.0, 0.3, 0.5, 0.7, 0.9]},
            "dataset.name": {"values": [args.dataset]},
            "gnn": {"values": [args.gnn]},
        },
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=sweep_function)


if __name__ == "__main__":
    main()
    wandb.finish()
