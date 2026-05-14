#!/bin/bash

datasets=("cora_graph")
shots=(1 3 5 10)

for dataset in "${datasets[@]}"; do
    for shot in "${shots[@]}"; do
        echo "Running sweep for dataset: $dataset, shot: $shot"
        python run_sweep.py --dataset $dataset --shot $shot
    done
done
