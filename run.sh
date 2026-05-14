#!/bin/bash

cfg_path=configs/cora/1-shot/tgfm.yaml

python pretrain.py --cfg $cfg_path
start_time=$(date +%s)
CUDA_LAUNCH_BLOCKING=1 python downstream.py --cfg $cfg_path --way 2
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Total time: ${total_time} seconds"
