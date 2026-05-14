# GFMate

**GFMate: Empowering Graph Foundation Models with Pre-training-agnostic Test-time Prompt Tuning** implementation and experiment scripts.

## Environment

```bash
conda env create -f environment.yml
conda activate gfmate
```

## Hardware

All experiments in this project (pretraining, downstream evaluation, backbone baselines, graph-level tasks, multi-seed runs, and W&B sweep trials) were run on **one** NVIDIA RTX A6000 (48 GB VRAM) GPU.

## Dataset

Place benchmark data under `datasets/` (see `datasets/readme.md`). If you use a release archive:

```bash
unzip datasets.zip -d datasets/
```

Layouts must match the paths and splits referenced in `configs/`.

## Reproduction (`configs/`)

Configuration files are organized under `configs/` by **dataset**. Each dataset occupies a dedicated subdirectory (for example, `cora/`, `texas/`, `arxiv-year/`, or graph-classification benchmarks such as `COX2/` and `cora_graph/`). Few-shot regimes are further separated into folders named `<k>-shot/` (for example, `1-shot/`, `3-shot/`). Within each such folder, one or more YAML files specify data-related options, model settings, and optimisation hyperparameters. To reproduce, run `pretrain.py` and `downstream.py` with the YAML you intend to use, or invoke the sweep drivers listed below.


## Scripts (for experiment driver)

| Script | Role |
|--------|------|
| **`run.sh`** | Minimal **single-run** template: `pretrain.py` then `downstream.py` for one `cfg_path` (please edit the YAML path inside the script). |
| **`run_sweep.sh`** / **`run_sweep_node.sh`** | Bash wrappers that call **`run_sweep.py`**, **node-level** few-shot experiments over `(dataset, shot)` grids; uses **`configs/<dataset>/<shot>-shot/*.yaml`** and W&B Sweep (see below). |
| **`run_sweep_base.sh`** | Calls **`run_sweep_base.py`** — **standard GNN backbones** (e.g. GCN / GAT / GraphSAGE) for controlled comparisons. |
| **`run_sweep_graph.sh`** | Calls **`run_sweep_graph.py`** — **graph classification** benchmarks (configs under graph dataset names, e.g. `COX2/`, `cora_graph/`). |
| **`run_sweep_pretrain.sh`** | Calls **`run_sweep_pretrain.py`** — **pretraining** sweeps across datasets. |
| **`run_all_seed.sh`** | **Multiple random seeds**: samples seeds, patches `seed:` into the chosen YAMLs, runs **`downstream.py`** repeatedly and aggregates mean / std into a CSV. |

Entrypoints (also used inside sweeps):

```bash
python pretrain.py --cfg configs/<dataset>/<N>-shot/tgfm.yaml
python downstream.py --cfg configs/<dataset>/<N>-shot/tgfm.yaml   # or .../gptt.yaml after a sweep / for a second preset
```

Edit the `*.sh` files to set `datasets=(...)`, `shots=(...)`, `gnns=(...)`, etc., before submitting or running locally.

### `tgfm.yaml` and `gptt.yaml`

These two basenames are a fixed convention in our repository. Both use the same configuration schema and are loaded with `cfg.merge_from_file`—`downstream.py` does not branch on the filename, only the YAML contents matter. The practical difference is how the sweep scripts use them:

| File | Role |
|------|------|
| **`tgfm.yaml`** | **TGFM** — the default configuration for the main test-time prompt tuning setup (see naming in code, e.g. comments in `utils/loss.py`). All `run_sweep*.py` drivers read this path as the base config (`config_path`). |
| **`gptt.yaml`** | **Sweep / trial output** — the `run_sweep*.py` drivers merge W&B search parameters into that base dict, `yaml.dump` the merged result to `gptt.yaml`, and launch **`downstream.py`** (and, where applicable, **`pretrain.py`**) with **`--cfg .../gptt.yaml`** so search trials do not overwrite `tgfm.yaml`. We keep `gptt` for historical layout consistency.

For **manual** runs you may pass either file to `--cfg`. The two YAMLs in the same `<k>-shot/` folder **may differ on purpose** (for example different `optim.epochs`). **Please treat `tgfm.yaml` as the stable reference and `gptt.yaml` as the file sweeps overwrite for hyperparameter tuning.**

## Wandb and sweeps

- **`run_sweep.py`**, **`run_sweep_base.py`**, **`run_sweep_graph.py`**, and **`run_sweep_pretrain.py`** each define **W&B Sweeps** (`wandb.sweep` / `wandb.agent`) so hyperparameters can be explored systematically; install/log in with the `wandb` CLI as usual. Sweep fields are read from `wandb.config` and written into the YAML used for each trial.
- **`downstream.py`** logs to W&B when the loaded config has **`wandb: true`** (default in code is off; set it in the YAML for single runs or for scripts like `run_all_seed.sh` if you want online metrics).
- **`pretrain.py`** does not initialise W&B in the current codebase, you should use the **pretrain sweep** driver for coordinated pretrain-side search.

The conda environment pins `wandb` (see `environment.yml`). Run directory `wandb/` is gitignored.

## Citation

If you use this repository or the GFMate method in research, please cite our paper:

```bibtex

```

## Acknowledgements

We thank the authors of **all related work**, including graph foundation models, graph prompting, semi-supervised and few-shot learning on graphs, and the open-source stacks this code builds on for their awesome work and released artifacts, which made this project possible.
