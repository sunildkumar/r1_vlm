#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

time uv run accelerate launch \
    --config_file train/single_gpu.yaml \
    --main_process_port 29701 \
    train/train.py
