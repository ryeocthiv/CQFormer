#!/usr/bin/env bash

CUDA_id=3

CUDA_VISIBLE_DEVICES=${CUDA_id} python train.py -d cifar10 --num_colors 16
CUDA_VISIBLE_DEVICES=${CUDA_id} python train.py -d cifar100 --num_colors 16
CUDA_VISIBLE_DEVICES=${CUDA_id} python train.py -d stl10 --num_colors 16
CUDA_VISIBLE_DEVICES=${CUDA_id} python train.py -d tiny200 --num_colors 16
