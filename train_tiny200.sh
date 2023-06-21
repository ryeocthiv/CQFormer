#!/usr/bin/env bash

# for ((i=0;i <=10;i++));do
#     CUDA_VISIBLE_DEVICES=3 python train.py -d cifar10 --num_colors 3 
# done

CUDA_VISIBLE_DEVICES=3 python train.py -d tiny200 --num_colors 3 
CUDA_VISIBLE_DEVICES=3 python train.py -d tiny200 --num_colors 5  
CUDA_VISIBLE_DEVICES=3 python train.py -d tiny200 --num_colors 6  
CUDA_VISIBLE_DEVICES=3 python train.py -d tiny200 --num_colors 7 
