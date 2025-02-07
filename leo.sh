#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0


time uv run python train/train.py \
    --load_from_local --local_path $HOME/data/textcodes
