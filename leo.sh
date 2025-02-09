#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=0
export SHOULD_PRINT_REWARD=true

time uv run python train/train.py \
    --load_from_local --local_path $HOME/data/textcodes \
    --max_completion_length 100 \
    --learning_rate 1e-4 \
    --warmup_steps 20 \
    --max_grad_norm 10.0 \
    --loss_magnifier 1.0 \
    --tool_return_nothing \
    --which_tool empty \
    --rewards justformat \
    --num_generations 10 \
    --gradient_accumulation_steps 10 \
    --run_name vlm-r1-textcodes-justformat-empty
