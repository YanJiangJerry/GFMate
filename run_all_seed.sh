#!/bin/bash

datasets=("wikics")
shot_sizes=("3-shot")
results_file="results_base_gfm_repeat.csv"
seeds=($(shuf -i 0-99999 -n 10))

echo "Dataset,Shot,Seed,Accuracy,Mean_Accuracy,Std" >> "$results_file"
for dataset in "${datasets[@]}"; do
    for shot in "${shot_sizes[@]}"; do
        accuracies=()
        for seed in "${seeds[@]}"; do
            cfg_path="configs/${dataset}/${shot}/tgfm.yaml"

            if grep -q "^seed:" "$cfg_path"; then
                sed -i "s/^seed: .*/seed: $seed/" "$cfg_path"
            else
                sed -i "1a seed: $seed" "$cfg_path"
            fi

            output=$(python downstream.py --cfg "$cfg_path")
            accuracy=$(echo "$output" | tail -n 1)
            accuracies+=("$accuracy")
            echo "$dataset,$shot,$seed,$accuracy" >> "$results_file"
        done
        mean_acc=$(printf "%s\n" "${accuracies[@]}" | awk '{sum+=$1} END {if (NR>0) print sum/NR; else print "0.000"}')
        std_acc=$(printf "%s\n" "${accuracies[@]}" | awk -v mean="$mean_acc" '{sum+=($1 - mean) * ($1 - mean)} END {if (NR>1) print sqrt(sum / (NR-1)); else print "0.000"}')

        echo "$dataset,$shot,Mean,Std,$mean_acc,$std_acc" >> "$results_file"
        echo "" >> "$results_file"
    done
done
