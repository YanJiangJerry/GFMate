#!/bin/bash

datasets=("citeseer")
gnns=("GCN")

for dataset in "${datasets[@]}"; do
    for gnn in "${gnns[@]}"; do
        echo "Running sweep for dataset: $dataset with GNN: $gnn"
        python run_sweep_base.py --dataset $dataset --gnn $gnn
    done
done
