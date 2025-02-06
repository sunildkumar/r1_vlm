#!/bin/bash

set -ex

cd $(dirname $0)/data

uv run python ./r1_dataset.py --local_path /home/leo/r1dev/_localdata/cocomath/ \
