# import wandb
# import yaml
# import os
# import subprocess
# import argparse

# # os.environ["WANDB_MODE"] = "offline"
# ## wandb sync


# def parse_args():
#     parser = argparse.ArgumentParser(description="Sweep Configuration")
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default="cora",
#         help="Dataset name (e.g., cora, amazon, etc.)",
#     )

#     parser.add_argument(
#         "--shot", type=int, default=10, help="Number of shots (e.g., 10, 5, etc.)"
#     )

#     return parser.parse_args()


# def load_base_config(config_path):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)


# def sweep_function():
#     with wandb.init():
#         config = wandb.config

#         ## only the parameter in the sweep config will be retrieved
#         dataset = config.get("dataset.name", "cora")
#         shot = config.get("num_shot", 1)
#         config_path = f"configs/{dataset}/{shot}-shot/tgfm.yaml"

#         base_config = load_base_config(config_path)

#         base_config.setdefault("optim", {})
#         base_config.setdefault("model", {})
#         # base_config.setdefault("wandb", True)
#         # base_config.setdefault("seed", 1)
#         # base_config.setdefault("debug_pre_train", True)
#         base_config.setdefault("debug", True)

#         base_config["optim"]["pre_train_lr"] = float(
#             config.get(
#                 "optim.pre_train_lr", base_config["optim"].get("pre_train_lr", 1e-4)
#             )
#         )
#         base_config["optim"]["pre_train_wd"] = float(
#             config.get(
#                 "optim.pre_train_wd", base_config["optim"].get("pre_train_wd", 0.0)
#             )
#         )
#         base_config["optim"]["pre_train_epochs"] = config.get(
#             "optim.pre_train_epochs", base_config["optim"].get("pre_train_epochs", 10)
#         )

#         base_config["model"]["alpha"] = float(
#             config.get("model.alpha", base_config["model"].get("alpha", 1.0))
#         )

#         base_config["model"]["hidden_dim"] = config.get(
#             "model.hidden_dim", base_config["model"].get("hidden_dim", 128)
#         )
#         base_config["model"]["num_layers"] = config.get(
#             "model.num_layers", base_config["model"].get("num_layers", 2)
#         )
#         base_config["model"]["dropout"] = float(
#             config.get("model.dropout", base_config["model"].get("dropout", 0.0))
#         )

#         base_config["optim"]["test_shot"] = config.get(
#             "optim.test_shot", base_config["optim"].get("test_shot", 1)
#         )
#         base_config["optim"]["layer"] = config.get(
#             "optim.layer", base_config["optim"].get("layer", 0)
#         )
#         base_config["optim"]["lr"] = float(
#             config.get("optim.lr", base_config["optim"].get("lr", 0.01))
#         )
#         base_config["optim"]["wd"] = float(
#             config.get("optim.wd", base_config["optim"].get("wd", 0.0001))
#         )
#         base_config["optim"]["epochs"] = config.get(
#             "optim.epochs", base_config["optim"].get("epochs", 100)
#         )

#         updated_cfg_path = f"configs/{dataset}/{shot}-shot/gptt.yaml"
#         os.makedirs(os.path.dirname(updated_cfg_path), exist_ok=True)
#         with open(updated_cfg_path, "w") as f:
#             yaml.dump(base_config, f)
#         os.system(f"python pretrain.py --cfg {updated_cfg_path}")
#         result = subprocess.run(
#             f"CUDA_LAUNCH_BLOCKING=1 python downstream.py --cfg {updated_cfg_path}",
#             shell=True,
#             check=True,
#             text=True,
#             capture_output=True,
#         )
#         second_last_line = result.stdout.strip().split("\n")[-3]
#         accuracy = float(result.stdout.strip().split("\n")[-1])
#         print(f"Dataset: {dataset} Shot: {shot} Acc: {second_last_line}")
#         wandb.log({"accuracy": accuracy})


# def main():
#     args = parse_args()
#     project_name = f"{args.dataset}_{args.shot}_gfm"
#     ## random, grid, greedy, bayes
#     sweep_config = {
#         "method": "bayes",
#         "run_cap": 500,
#         "name": project_name,
#         "project": project_name,
#         "metric": {"name": "accuracy", "goal": "maximize"},
#         "parameters": {
#             # cfg.optim.pre_train_delta = 0.1
#             # cfg.optim.pre_train_lr = 1e-4
#             # cfg.optim.pre_train_wd = 0.0
#             # cfg.optim.pre_train_epochs = 20
#             # Hidden layer dim
#             # cfg.model.hidden_dim = 256
#             # # Layer number
#             # cfg.model.num_layers = 3
#             # # Dropout rate
#             # cfg.model.dropout = 0.5
#             "model.alpha": {
#                 "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#             },
#             "optim.pre_train_lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
#             "optim.pre_train_wd": {"values": [0.0, 1e-4, 1e-3]},
#             "optim.pre_train_epochs": {"values": [5, 10, 20, 50]},
#             "model.hidden_dim": {"values": [64, 128, 256]},
#             "model.num_layers": {"values": [2, 3, 4, 5]},
#             "model.dropout": {"values": [0.0, 0.3, 0.5, 0.7]},
#             # cfg.optim.test_shot = 1
#             # cfg.optim.wd = 5e-4
#             # cfg.optim.lr = 1e-4
#             # cfg.optim.epochs = 100
#             ## PromptTuning
#             "optim.lr": {"values": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
#             "optim.wd": {"values": [0, 1e-4, 1e-3, 1e-2]},
#             "optim.epochs": {"values": [10, 20, 50, 100]},
#             "optim.prompt_tau": {
#                 "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#             },
#             "optim.test_shot": {"values": [1, 3, 5, 10, 20, 50, 100]},
#             "optim.layer": {"values": [0, 1, 2]},
#             ## Hyperparameters to pass in sweep function
#             "dataset.name": {"values": [args.dataset]},
#             "num_shot": {"values": [args.shot]},
#         },
#     }
#     ## Should not pass in any else parameter than the sweep_config
#     sweep_id = wandb.sweep(sweep_config)
#     wandb.agent(sweep_id, function=sweep_function)


# if __name__ == "__main__":
#     main()
#     wandb.finish()


import wandb
import yaml
import os
import subprocess
import argparse

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
        "--shot", type=int, default=10, help="Number of shots (e.g., 10, 5, etc.)"
    )
    return parser.parse_args()


def load_base_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sweep_function():
    with wandb.init():
        config = wandb.config

        ## only the parameter in the sweep config will be retrieved
        dataset = config.get("dataset.name", "cora")
        shot = config.get("num_shot", 10)
        # config_path = f"configs/{dataset}/{shot}-shot/tgfm.yaml"
        # updated_cfg_path = f"configs/{dataset}/{shot}-shot/gptt.yaml"
        config_path = f"configs/{dataset}/{shot}-shot/tgfm.yaml"
        updated_cfg_path = f"configs/{dataset}/{shot}-shot/gptt.yaml"

        base_config = load_base_config(config_path)
        base_config.setdefault("optim", {})
        base_config.setdefault("model", {})
        # base_config.setdefault("wandb", True)
        # base_config.setdefault("seed", 1)
        # base_config.setdefault("full_shot", True)
        # base_config.setdefault("ablation", True)

        # base_config["optim"]["tau"] = float(
        #     config.get("optim.tau", base_config["optim"].get("tau", 0.1))
        # )
        # base_config["optim"]["delta"] = float(
        #     config.get("optim.delta", base_config["optim"].get("delta", 0.0))
        # )
        # base_config["optim"]["prompt_tau"] = float(
        #     config.get("optim.prompt_tau", base_config["optim"].get("prompt_tau", 0.1))
        # )
        # base_config["optim"]["prompt_delta"] = float(
        #     config.get(
        #         "optim.prompt_delta", base_config["optim"].get("prompt_delta", 0.0)
        #     )
        # )
        # base_config["optim"]["test_tau"] = float(config.get(
        #     "optim.test_tau", base_config["optim"].get("test_tau", 0.1)
        # ))
        # base_config["optim"]["test_delta"] = float(config.get(
        #     "optim.test_delta", base_config["optim"].get("test_delta", 0.0)
        # ))
        base_config["optim"]["test_shot"] = config.get(
            "optim.test_shot", base_config["optim"].get("test_shot", 1)
        )
        # base_config["optim"]["layer"] = config.get(
        #     "optim.layer", base_config["optim"].get("layer", 0)
        # )
        # base_config["optim"]["test_epochs"] = config.get(
        #     "optim.test_epochs", base_config["optim"].get("test_epochs", 10)
        # )

        # base_config["model"]["alpha"] = float(
        #     config.get("model.alpha", base_config["model"].get("alpha", 1.0))
        # )

        # base_config["model"]["hidden_dim"] = config.get(
        #     "optim.test_shot", base_config["optim"].get("test_shot", 128)
        # )
        # base_config["model"]["num_layers"] = config.get(
        #     "optim.test_shot", base_config["optim"].get("test_shot", 2)
        # )
        # base_config["optim"]["pre_train_epochs"] = config.get(
        #     "optim.test_shot", base_config["optim"].get("test_shot", 10)
        # )

        # cfg.optim.lr = 1e-4
        # cfg.optim.epochs = 500
        # cfg.optim.prompt_tau = 0.1

        base_config["optim"]["lr"] = float(
            config.get("optim.lr", base_config["optim"].get("lr", 0.01))
        )
        base_config["optim"]["wd"] = float(
            config.get("optim.wd", base_config["optim"].get("wd", 0.0001))
        )
        base_config["optim"]["epochs"] = config.get(
            "optim.epochs", base_config["optim"].get("epochs", 100)
        )
        base_config["optim"]["prompt_delta"] = float(
            config.get(
                "optim.prompt_delta", base_config["optim"].get("prompt_delta", 0.1)
            )
        )
        base_config["optim"]["prompt_tau"] = float(
            config.get("optim.prompt_tau", base_config["optim"].get("prompt_tau", 0.1))
        )

        os.makedirs(os.path.dirname(updated_cfg_path), exist_ok=True)
        with open(updated_cfg_path, "w") as f:
            yaml.dump(base_config, f)
        # os.system(f"python pretrain.py --cfg {updated_cfg_path}")
        result = subprocess.run(
            f"CUDA_LAUNCH_BLOCKING=1 python downstream.py --cfg {updated_cfg_path}",
            shell=True,
            check=True,
            text=True,
            capture_output=True,
        )
        second_last_line = result.stdout.strip().split("\n")[-3]
        accuracy = float(result.stdout.strip().split("\n")[-1])
        print(f"Dataset: {dataset} Shot: {shot} Acc: {second_last_line}")
        wandb.log({"accuracy": accuracy})


def main():
    args = parse_args()
    project_name = f"{args.dataset}_{args.shot}_gfm"
    ## random, grid, greedy, bayes
    sweep_config = {
        "method": "bayes",
        "run_cap": 100,
        "name": project_name,
        "project": project_name,
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            # "optim.tau": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
            # "optim.delta": {
            #     "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # },
            # "optim.prompt_tau": {
            #     "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # },
            # "optim.prompt_delta": {
            #     "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # },
            # "optim.test_tau": {
            #     "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # },
            # "optim.test_delta": {
            #     "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # },
            "optim.test_shot": {"values": [1, 3, 5, 10, 20, 50, 100]},
            # "optim.layer": {"values": [0, 1, 2]},
            # "optim.test_epochs": {
            #     "values": [10, 50, 100]
            # },
            # "model.alpha": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            # "model.hidden_dim": {
            #     "values": [64, 128, 256, 512]
            # },
            # "model.num_layers": {
            #     "values": [2, 3, 4, 5]
            # },
            # "optim.pre_train_epochs": {
            #     "values": [10, 50, 100]
            # },
            "optim.lr": {"values": [1e-4, 1e-3, 1e-2, 1e-1]},
            "optim.wd": {"values": [0.0, 1e-4, 1e-3, 1e-2, 1e-1]},
            "optim.epochs": {"values": [10, 20, 50, 100, 200, 500]},
            "optim.prompt_tau": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            },
            "optim.prompt_delta": {
                "values": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ]
            },
            # "optim.prompt_beta": {
            #     "values": [
            #         0.1,
            #         0.2,
            #         0.3,
            #         0.4,
            #         0.5,
            #         0.6,
            #         0.7,
            #         0.8,
            #         0.9,
            #         1.0,
            #     ]
            # },
            # cfg.optim.test_shot = 1
            # cfg.optim.wd = 5e-4
            # cfg.optim.lr = 1e-4
            # cfg.optim.epochs = 100
            ## Hyperparameters to pass in sweep function
            "dataset.name": {"values": [args.dataset]},
            "num_shot": {"values": [args.shot]},
        },
    }
    ## Should not pass in any else parameter than the sweep_config
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=sweep_function)


if __name__ == "__main__":
    main()
    wandb.finish()
