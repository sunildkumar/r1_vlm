#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=0
export SHOULD_PRINT_REWARD=true

time uv run python train/train.py \
    --load_from_local --local_path $HOME/data/textcodes \
    --max_completion_length 150 \
    --learning_rate 1e-5 \
    --warmup_steps 20 \
    --max_grad_norm 3.0 \
    --loss_magnifier 1.0 \
    --tool_return_nothing \
    --rewards toolformat \
    --num_generations 6 \
    --gradient_accumulation_steps 4 \
    --run_name vlm-r1-textcodes-toolformat-nothing
