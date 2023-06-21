#!/usr/bin/env bash

# for ((i=0;i <=10;i++));do
#     CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 3 
# done

CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 3  --lr 0.01
CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 5  --lr 0.01
CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 6  --lr 0.01
CUDA_VISIBLE_DEVICES=0 python train.py -d cifar10 --num_colors 7 --lr 0.01
