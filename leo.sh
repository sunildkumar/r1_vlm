#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=1
export SHOULD_PRINT_REWARD=true
time uv run python train/train.py \
    --load_from_local --local_path $HOME/data/textcodes \
    --max_completion_length 128 \
    --learning_rate 1e-5 \
    --rewards justtool \
    --num_generations 7 
