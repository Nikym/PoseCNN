#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/demo.py --gpu 0 \
  --network vgg16_convs \
  --model output/fat/lov_train/fat_data_fat_iter_99908.ckpt \
  --imdb lov_keyframe \
  --cfg experiments/cfgs/fat_data.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl
