#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0


time uv run python train/train.py \
    --load_from_local --local_path /home/leo/r1dev/_localdata/cocomath/

exit

#time uv run accelerate launch \
    #--config_file train/single_gpu.yaml \
    #--main_process_port 29701 \
    #train/train.py
