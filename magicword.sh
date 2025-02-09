#!/bin/bash
# THis is something of a reference job - it converges to perfect reward in ~15 steps.

set -ex

export CUDA_VISIBLE_DEVICES=0

export VERBOSE=0
export SHOULD_PRINT_REWARD=true

time uv run python train/train.py \
    --load_from_local --local_path $HOME/data/magicword \
    --max_completion_length 100 \
    --learning_rate 1e-4 \
    --max_grad_norm 10.0 \
    --loss_magnifier 1.0 \
    --rewards magicword \
    --num_generations 8 \
    --gradient_accumulation_steps 4 \
    --run_name magicword-magloss
