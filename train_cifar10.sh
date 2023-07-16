#!/usr/bin/env bash

# for ((i=0;i <=10;i++));do
#     CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 3 
# done

CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 2
CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 4
CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 8 
CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 16
