#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=0
export SHOULD_PRINT_REWARD=true

time uv run accelerate launch \
    --config_file train/bs1.yaml \
    --main_process_port 29701 \
    train/train.py
    --load_from_local --local_path $HOME/data/magicword \
    --max_completion_length 64 \
    --learning_rate 1e-6 \
    --rewards magicword \
    --num_generations 3 \
    --run_name magicword 
