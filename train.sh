#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=3 python train.py -d cifar10  --num_colors 2
CUDA_VISIBLE_DEVICES=2 python train.py -d cifar10  --num_colors 4
CUDA_VISIBLE_DEVICES=1 python train.py -d cifar10  --num_colors 8
CUDA_VISIBLE_DEVICES=4 python train.py -d cifar10  --num_colors 16
CUDA_VISIBLE_DEVICES=5 python train.py -d cifar10  --num_colors 32
CUDA_VISIBLE_DEVICES=6 python train.py -d cifar10  --num_colors 64

CUDA_VISIBLE_DEVICES=3 python train.py -d cifar100  --num_colors 2
CUDA_VISIBLE_DEVICES=2 python train.py -d cifar100  --num_colors 4
CUDA_VISIBLE_DEVICES=1 python train.py -d cifar100  --num_colors 8
CUDA_VISIBLE_DEVICES=0 python train.py -d cifar100  --num_colors 16
CUDA_VISIBLE_DEVICES=5 python train.py -d cifar100  --num_colors 32
CUDA_VISIBLE_DEVICES=7 python train.py -d cifar100  --num_colors 64


CUDA_VISIBLE_DEVICES=3 python train.py -d stl10  --num_colors 2
CUDA_VISIBLE_DEVICES=2 python train.py -d stl10  --num_colors 4
CUDA_VISIBLE_DEVICES=1 python train.py -d stl10  --num_colors 8
CUDA_VISIBLE_DEVICES=0 python train.py -d stl10  --num_colors 16
CUDA_VISIBLE_DEVICES=5 python train.py -d stl10  --num_colors 32
CUDA_VISIBLE_DEVICES=7 python train.py -d stl10  --num_colors 64

# CUDA_VISIBLE_DEVICES=3 python train.py -d tiny200  --num_colors 2
# CUDA_VISIBLE_DEVICES=6 python train.py -d tiny200  --num_colors 4
# CUDA_VISIBLE_DEVICES=0 python train.py -d tiny200 --num_colors 8
# CUDA_VISIBLE_DEVICES=1 python train.py -d tiny200  --num_colors 16
# CUDA_VISIBLE_DEVICES=2 python train.py -d tiny200  --num_colors 32
# CUDA_VISIBLE_DEVICES=7 python train.py -d tiny200  --num_colors 64

#CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10  --num_colors 2
#CUDA_VISIBLE_DEVICES=1 python train.py -d cifar10  --num_colors 3
#CUDA_VISIBLE_DEVICES=2 python train.py -d cifar10  --num_colors 4
#CUDA_VISIBLE_DEVICES=3 python train.py -d cifar10  --num_colors 5
#CUDA_VISIBLE_DEVICES=4 python train.py -d cifar10  --num_colors 6
#CUDA_VISIBLE_DEVICES=5 python train.py -d cifar10  --num_colors 7
#CUDA_VISIBLE_DEVICES=5 python train.py -d cifar10  --num_colors 8

