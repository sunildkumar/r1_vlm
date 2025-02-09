#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=0
export SHOULD_PRINT_REWARD=true

time uv run python train/train.py \
    --max_completion_length 512 \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --tool_return_nothing
