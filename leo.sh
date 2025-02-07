#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=0
export SHOULD_PRINT_REWARD=true

time uv run python train/train.py \
    --load_from_local --local_path $HOME/data/textcodes \
    --max_completion_length 128 \
    --learning_rate 9e-5 \
    --rewards justtool \
    --num_generations 5 \
    --tool_return_nothing
