#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=0
export SHOULD_PRINT_REWARD=true

time uv run python train/train.py \
    --load_from_local --local_path $HOME/data/magicword \
    --max_completion_length 128 \
    --learning_rate 1e-6 \
    --rewards magicword \
    --num_generations 3 \
    --run_name magicword 
