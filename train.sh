#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10  --num_colors 2 --num_s 1
CUDA_VISIBLE_DEVICES=1 python train.py -d cifar10  --num_colors 2 --num_s 2
CUDA_VISIBLE_DEVICES=2 python train.py -d cifar10  --num_colors 4 --num_s 2
CUDA_VISIBLE_DEVICES=3 python train.py -d cifar10  --num_colors 4 --num_s 4
CUDA_VISIBLE_DEVICES=4 python train.py -d cifar10  --num_colors 8 --num_s 4
CUDA_VISIBLE_DEVICES=5 python train.py -d cifar10  --num_colors 8 --num_s 8


