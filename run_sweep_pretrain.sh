#!/bin/bash

datasets=("texas" "cornell" "wisconsin" "chameleon" "squirrel" "cora" "citeseer" "photo" "arxiv-year")

for dataset in "${datasets[@]}"; do
    echo "Running sweep for dataset: $dataset"
    python run_sweep_pretrain.py --dataset $dataset
done
