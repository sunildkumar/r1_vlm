#!/bin/bash

set -ex

cd $(dirname $0)/mnisttool

time uv run python ./textonly_mnist.py --save_path $HOME/data/magicword
