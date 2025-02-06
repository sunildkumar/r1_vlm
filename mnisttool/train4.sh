#!/bin/bash

uv run python train_vision_tool.py \
    --mixed_precision bf16 \
    --dataset_name your_dataset_name \
    --max_steps 1000 \
    --save_steps 500


